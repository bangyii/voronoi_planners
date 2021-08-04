#include <voronoi_planner_lib/voronoi_path.h>
#include <voronoi_planner_lib/profiler.h>
#include <iostream>
#include <algorithm>
#include <exception>
#include <future>
#include <thread>
#include <functional>
#include <cmath>

namespace voronoi_path
{
    VoronoiPath::VoronoiPath() : ebo(map_ptr)
    {
    }

    void VoronoiPath::setLocalVertices(const std::vector<GraphNode> &vertices)
    {
        local_vertices = vertices;
    }

    uint32_t VoronoiPath::getUniqueID()
    {
        static uint32_t id = 1;
        return id++;
    }

    void VoronoiPath::updateEBandParams()
    {
        ebo.num_optim_iterations_ = num_optim_iterations_;
        ebo.internal_force_gain_ = internal_force_gain_;
        ebo.external_force_gain_ = external_force_gain_;
        ebo.tiny_bubble_distance_ = tiny_bubble_distance_;
        ebo.tiny_bubble_expansion_ = tiny_bubble_expansion_;
        ebo.min_bubble_overlap_ = min_bubble_overlap_;
        ebo.max_recursion_depth_approx_equi_ = max_recursion_depth_approx_equi_;
        ebo.equilibrium_relative_overshoot_ = equilibrium_relative_overshoot_;
        ebo.significant_force_ = significant_force_;
        ebo.costmap_weight_ = costmap_weight_;
    }

    std::vector<std::complex<double>> VoronoiPath::findObstacleCentroids()
    {
        if (map_ptr->data.size() != 0)
        {
            Profiler profiler;
            cv::Mat cv_map = cv::Mat(map_ptr->data).reshape(0, map_ptr->height);
            cv_map.convertTo(cv_map, CV_8UC1);

            //Reset all centroid variables
            centers.clear();
            obs_coeff.clear();

            //Downscale to increase contour finding speed
            cv::resize(cv_map, cv_map, cv::Size(), open_cv_scale, open_cv_scale, cv::INTER_AREA);

            //Flip and transpose because image from map_server and actual map orientation is different
            cv::flip(cv_map, cv_map, 1);
            cv::transpose(cv_map, cv_map);
            cv::flip(cv_map, cv_map, 1);
            if (print_timings)
                profiler.print("findObstacleCentroids copy map data");
            
            //Erode by 3px to increase chances of contour point lying in collision zone
            cv::Mat structure_elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::erode(cv_map, cv_map, structure_elem);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::Canny(cv_map, cv_map, 50, 150, 3);
            cv::findContours(cv_map, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
            centers = std::vector<std::complex<double>>(contours.size());

            for (int i = 0; i < contours.size(); ++i)
            {
                //Find part of contour that lies on the inflation zone
                for (int j = 0; j < contours[i].size(); ++j)
                {
                    centers[i] = std::complex<double>(map_ptr->width - contours[i][j].y / open_cv_scale, map_ptr->height - contours[i][j].x / open_cv_scale);
                    if (map_ptr->data[floor(centers[i].real()) + floor(centers[i].imag()) * map_ptr->width] > collision_threshold)
                        break;

                    if (j == contours[i].size() - 1)
                    {
                        // std::cout << "WARN: Could not find point on contour " << j << " which lies on the obstacle's inflation zone, path finding may not produce";
                        // std::cout << " paths that are strictly in different homotopy classes\n ";
                    }
                }
            }

            double a = (centers.size() - 1) / 2.0;
            double b = a;
            obs_coeff.resize(centers.size(), std::complex<double>(1, 1));

            std::complex<double> from_begin(1, 1);
            std::complex<double> from_end(1, 1);

            //Scale down centers used for calculating homotopy coefficients to prevent overflow of double
            double max = map_ptr->width > map_ptr->height ? map_ptr->width : map_ptr->height;
            auto scaled_centers(centers);
            for (auto &centers : scaled_centers)
                centers /= max;

            for (int i = 0; i < scaled_centers.size(); ++i)
            {
                obs_coeff[i] *= from_begin;
                from_begin *= scaled_centers[i];

                obs_coeff[scaled_centers.size() - i - 1] *= from_end;
                from_end *= scaled_centers[scaled_centers.size() - i - 1];
                if (std::isnan(obs_coeff[i].real()) || std::isnan(obs_coeff[i].imag()))
                {
                    std::cout << "Obstacle coefficients calculation produced NaN value, unique homotopy class exploration";
                    std::cout << "will not work properly\n ";
                }
            }

            if (print_timings)
                profiler.print("findObstacleCentroids find contour");
        }

        return centers;
    }

    std::vector<jcv_point> VoronoiPath::fillOccupancyVector(const int &start_index, const int &num_pixels)
    {
        std::vector<jcv_point> points_vec;
        for (int i = start_index; i < start_index + num_pixels; i += (pixels_to_skip + 1))
        {
            //Occupied
            if (map_ptr->data[i] >= occupancy_threshold)
            {
                jcv_point temp_point;
                temp_point.x = i % map_ptr->width;
                temp_point.y = static_cast<int>(i / map_ptr->width);

                points_vec.push_back(temp_point);
            }
        }

        return points_vec;
    }

    bool VoronoiPath::mapToGraph(Map *map_ptr_)
    {
        //Lock mutex to ensure adj_list is not being used
        Profiler complete_profiler, section_profiler;
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        map_ptr = map_ptr_;

        if (print_timings)
            section_profiler.print("mapToGraph lock duration");

        //Get centroids after map has been updated
        findObstacleCentroids();

        int size = map_ptr->data.size();
        if (size == 0)
            return false;

        //Set bottom left and top right for use during homotopy check
        BL = std::complex<double>(0, 0);
        TR = std::complex<double>(map_ptr->width - 1, map_ptr->height - 1);

        // Loop through map to find occupied cells
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<std::vector<jcv_point>>> future_vector;
        future_vector.reserve(num_threads - 1);

        int num_pixels = floor(size / num_threads);
        int start_pixel = 0;

        for (int i = 0; i < num_threads - 1; ++i)
        {
            start_pixel = i * num_pixels;
            future_vector.emplace_back(std::async(std::launch::async, &VoronoiPath::fillOccupancyVector, this, start_pixel, num_pixels));
        }

        //For last thread, take all remaining pixels
        //This current thread is the nth thread
        std::vector<jcv_point> points_vec;
        points_vec = fillOccupancyVector((num_threads - 1) * num_pixels, size - num_pixels * (num_threads - 1));

        for (int i = 0; i < future_vector.size(); ++i)
        {
            try
            {
                future_vector[i].wait();
                std::vector<jcv_point> temp_vec = future_vector[i].get();
                points_vec.insert(points_vec.end(), temp_vec.begin(), temp_vec.end());
            }
            catch (const std::exception &e)
            {
                std::cout << "Exception occurred with future, " << e.what() << std::endl;
                return false;
            }
        }

        //Add vertices that correspond to local costmap 4 corners
        for (int i = 0; i < local_vertices.size(); ++i)
        {
            jcv_point temp_point;
            temp_point.x = local_vertices[i].x;
            temp_point.y = local_vertices[i].y;
            points_vec.push_back(temp_point);
        }

        int occupied_points = points_vec.size();
        jcv_point *points = (jcv_point *)malloc(occupied_points * sizeof(jcv_point));

        if (!points)
        {
            std::cout << "Failed to allocate memory for points array" << std::endl;
            return false;
        }

        for (int i = 0; i < occupied_points; ++i)
            points[i] = points_vec[i];

        if (print_timings)
            section_profiler.print("mapToGraph loop map points");

        //Set the minimum and maximum bounds for voronoi diagram. Follows size of map
        jcv_rect rect;
        rect.min.x = 0;
        rect.min.y = 0;
        rect.max.x = map_ptr->width - 1;
        rect.max.y = map_ptr->height - 1;

        jcv_diagram diagram;
        memset(&diagram, 0, sizeof(jcv_diagram));

        //Tried diagram generation in another thread, does not help
        jcv_diagram_generate(occupied_points, points, &rect, 0, &diagram);

        //Get edges from voronoi diagram
        std::vector<const jcv_edge *> edge_vector;
        const jcv_edge *edges = jcv_diagram_get_edges(&diagram);
        while (edges)
        {
            edge_vector.push_back(edges);
            edges = jcv_diagram_get_next_edge(edges);
        }

        if (print_timings)
            section_profiler.print("mapToGraph generating edges");

        //Remove edge vertices that are in obtacle
        removeObstacleVertices(edge_vector);

        //Remove edges that pass through obstacle
        removeCollisionEdges(edge_vector);

        if (print_timings)
            section_profiler.print("mapToGraph clearing edges");

        //Convert edges to adjacency list
        edgesToAdjacency(edge_vector);

        if (print_timings)
        {
            section_profiler.print("mapToGraph convert edges to adjacency");
            complete_profiler.print("mapToGraph total time");
        }

        jcv_diagram_free(&diagram);
        free(points);
        return true;
    }

    bool VoronoiPath::edgesToAdjacency(const std::vector<const jcv_edge *> &edge_vector)
    {
        Profiler complete_profiler, section_profiler;
        //Reset all variables
        adj_list.clear();
        node_inf.clear();

        std::unordered_map<uint32_t, int> hash_index_map;
        for (int i = 0; i < edge_vector.size(); ++i)
        {
            //Get hash for both vertices of the current edge
            std::vector<uint32_t> hash_vec = {hash(edge_vector[i]->pos[0].x, edge_vector[i]->pos[0].y),
                                              hash(edge_vector[i]->pos[1].x, edge_vector[i]->pos[1].y)};
            int node_index[] = {-1, -1};

            //Check if each node is already in the map
            for (int j = 0; j < 2; ++j)
            {
                auto node_it = hash_index_map.find(hash_vec[j]);

                //Node already exists
                if (node_it != hash_index_map.end())
                    node_index[j] = node_it->second;

                //Node doesn't exist, add new node to adjacency list & info vector, and respective hash and node index to map
                else
                {
                    node_index[j] = adj_list.size();
                    node_inf.emplace_back(edge_vector[i]->pos[j].x, edge_vector[i]->pos[j].y);
                    adj_list.push_back(std::vector<int>());
                    hash_index_map.insert(std::pair<uint32_t, int>(hash_vec[j], node_index[j]));
                }
            }

            //Once both node indices are found, add edge between the two nodes if they aren't the same node
            if (node_index[0] != node_index[1])
            {
                adj_list[node_index[0]].push_back(node_index[1]);
                adj_list[node_index[1]].push_back(node_index[0]);
            }
        }

        if(print_timings)
            section_profiler.print("edgesToAdjacency hash time");

        //Connect single edges to nearby node if <= node_connection_threshold_pix pixel distance
        std::vector<int> unconnected_nodes;
        int threshold = pow(node_connection_threshold_pix, 2);
        for (int node_num = 0; node_num < adj_list.size(); ++node_num)
        {
            //Singly connected node
            if (adj_list[node_num].size() == 1)
            {
                //Check through all node_inf to see if there are any within distance threshold
                for (int j = 0; j < node_inf.size(); ++j)
                {
                    //If the node being checked is itself or is already connected to the checked node, new_adj_list[i] only has 1 element
                    if (j == node_num || adj_list[node_num].back() == j)
                        continue;

                    double dist = pow(node_inf[j].x - node_inf[node_num].x, 2) + pow(node_inf[j].y - node_inf[node_num].y, 2);
                    if (dist <= threshold)
                    {
                        adj_list[node_num].push_back(j);
                        adj_list[j].push_back(node_num);

                        //Check if this connection creates a cycle within N nodes threshold
                        std::vector<int> visited_list;
                        if(hasCycle(node_num, 0, visited_list))
                        {
                            adj_list[node_num].pop_back();
                            adj_list[j].pop_back();
                        }

                        break;
                    }

                    //Remember nodes that were unconnected to trim later
                    if(j == node_inf.size() - 1)
                        unconnected_nodes.push_back(node_num);
                }
            }
        }

        if(print_timings)
            section_profiler.print("edgesToAdjacency connect single nodes time");

        //Loop through all nodes that were unable to be connected for trimming
        auto new_adj_list = adj_list;
        double thresh = sqrt(lonely_branch_dist_threshold) / map_ptr->resolution;
        for(const auto &node_num : unconnected_nodes)
            removeExcessBranch(new_adj_list, thresh, node_num);

        if(print_timings)
            section_profiler.print("edgesToAdjacency remove excess branch time");

        adj_list = std::move(new_adj_list);
        num_nodes = adj_list.size();

        if(print_timings)
            complete_profiler.print("edgesToAdjacency total time");
        return true;
    }
    
    bool VoronoiPath::hasCycle(int cur_node, int cur_depth, std::vector<int> &visited_list, int prev)
    {
        if(cur_depth > node_depth_threshold)
            return false;

        std::vector<int> open_list;
        visited_list.push_back(cur_node);

        //Add adjacent nodes to open list unless it is previous node
        for(int i = 0; i < adj_list[cur_node].size(); ++i)
        {
            if(prev != adj_list[cur_node][i])
            {
                //If node to be added is already in visited list, cycle found
                if(std::find(visited_list.begin(), visited_list.end(), adj_list[cur_node][i]) != visited_list.end())
                    return true;

                open_list.push_back(adj_list[cur_node][i]);
            }
        }

        while(!open_list.empty())
        {
            if(hasCycle(open_list.back(), cur_depth + 1, visited_list, cur_node))
                return true;
            
            open_list.pop_back();
        }

        return false;
    }

    std::vector<std::vector<int>> VoronoiPath::getAdjList()
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        return adj_list;
    }

    std::vector<GraphNode> VoronoiPath::getNodeInfo()
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        return node_inf;
    }

    std::vector<std::pair<double, int>> VoronoiPath::getSortedNodeList(GraphNode position)
    {
        sorted_node_list.clear();
        double min_start_dist = std::numeric_limits<double>::infinity();

        //Store list of distances to each node from current position
        for (int i = 0; i < num_nodes; ++i)
            sorted_node_list.emplace_back(pow(node_inf[i].x - position.x, 2) + pow(node_inf[i].y - position.y, 2), i);

        //Sort list of nodes and distance
        sort(sorted_node_list.begin(), sorted_node_list.end(), [](std::pair<double, int> &left, std::pair<double, int> &right) {
            return left.first < right.first;
        });
        return sorted_node_list;
    }

    bool VoronoiPath::getObstacleCentroids(std::vector<GraphNode> &centroids)
    {
        centroids.reserve(centers.size());
        for (const auto &elem : centers)
            centroids.emplace_back(elem.real(), elem.imag());

        return true;
    }

    std::vector<double> VoronoiPath::getAllPathCosts()
    {
        return previous_path_costs;
    }

    bool VoronoiPath::getEdges(std::vector<GraphNode> &edges)
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);

        // Total number is unknown, reserve minimum amount needed
        edges.reserve(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            for (int j = 0; j < adj_list[i].size(); ++j)
            {
                edges.emplace_back(node_inf[i].x, node_inf[i].y);
                edges.emplace_back(node_inf[adj_list[i][j]].x, node_inf[adj_list[i][j]].y);
            }
        }

        return true;
    }

    bool VoronoiPath::getDisconnectedNodes(std::vector<GraphNode> &nodes)
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        for (int i = 0; i < num_nodes; ++i)
        {
            //If the node is only connected on one side
            if (adj_list[i].size() == 1)
                nodes.emplace_back(node_inf[i].x, node_inf[i].y);
        }

        return true;
    }

    void VoronoiPath::printEdges()
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        for (int i = 0; i < num_nodes; ++i)
        {
            for (int j = 0; j < adj_list[i].size(); ++j)
            {
                std::cout << node_inf[i].x << "\n";
                std::cout << node_inf[i].y << "\n";
                std::cout << node_inf[adj_list[i][j]].x << "\n";
                std::cout << node_inf[adj_list[i][j]].y << "\n";
            }
        }
        std::cout << std::endl;
    }

    uint32_t VoronoiPath::hash(const double &x, const double &y)
    {
        uint32_t hashed_int = static_cast<uint32_t>((static_cast<uint16_t>(x) << 16) ^ static_cast<uint16_t>(y));
        return hashed_int;
    }

    bool VoronoiPath::interpolateContractPaths(std::vector<Path> &paths)
    {
        //Increase resolution of paths by interpolation before contracting to give smoother result
        interpolatePaths(paths, path_waypoint_sep);
        if(!use_elastic_band)
        {
            for (auto &path : paths)
            {
                contractPath(path.path);

                if(!backtrack_paths)
                    findStuckVertex(path.path);
            }
        }

        //Use eband
        else
        {
            for (int i = 0; i < paths.size(); ++i)
            {
                std::vector<geometry_msgs::PoseStamped> global_plan;
                std::vector<GraphNode> new_path;

                //Convert to map frame plan
                for(const auto &pose : paths[i].path)
                {
                    geometry_msgs::PoseStamped new_pose;
                    new_pose.header.frame_id = map_ptr->frame_id;
                    map_ptr->mapToWorld(pose.x, pose.y, new_pose.pose.position.x, new_pose.pose.position.y);
                    new_pose.pose.orientation.w = 1.0;
                    global_plan.push_back(std::move(new_pose));
                }

                map_ptr->robot_radius = robot_radius;
                ebo.setPlan(global_plan, map_ptr);
                ebo.optimizeBand();
                ebo.getPlan(global_plan);

                for(const auto &pose : global_plan)
                {
                    GraphNode new_node;
                    map_ptr->worldToMap(pose.pose.position.x, pose.pose.position.y, new_node.x, new_node.y);
                    new_path.push_back(std::move(new_node));
                }

                paths[i].path = std::move(new_path);
            }
        }

        return true;
    }

    bool VoronoiPath::findStuckVertex(std::vector<GraphNode> &path)
    {
        // for(int j = 1; j < path.size() - 1; ++j)
        // {
        //     //Check if the prev point, current point, and next point form a vertex with angle that exceeds threshold
        //     if(j - 1 >= 0 && j + 1 < path.size())
        //     {
        //         auto v1 = path[j-1] - path[j];
        //         auto v2 = path[j+1] - path[j];
        //         double v1_mag = sqrt(pow(v1.x, 2) + pow(v1.y, 2));
        //         double v2_mag = sqrt(pow(v2.x, 2) + pow(v2.y, 2));
        //         double dot = v1.x * v2.x + v1.y * v2.y;
        //         double angle = acos(dot / (v1_mag * v2_mag));
        //         if(fabs(angle) < 65/180.0*M_PI)
        //             std::cout << "Angle of " << j << " is " << angle << " v1 " << v1.x << ", " << v1.y << " v2 " << v2.x << ", " << v2.y << "\n";
        //         if(fabs(angle) < path_vertex_angle_threshold/180.0*M_PI)
        //         {
        //             j = std::distance(path.begin(), path.erase(path.begin() + j)) - 1;
        //             if(j > 1 && j < path.size())
        //                 std::cout << "Removed point========================== " << j << "\n";
        //         }
                    
        //     }
        // }

        //Calculate the cumulative angle turned by 3 consecutive vectors formed by 4 vertices
        //i indicates the last vertex of the 3 vectors that is being evaluated
        for(int i = 3; i < path.size(); ++i)
        {
            std::vector<double> vector_angles;
            vector_angles.resize(3);

            //j indicates the end vertex of current vector being evaluated
            double total_mag = 0;
            for(int j = 3; j > 0; --j)
            {
                auto vec = path[i - j + 1] - path[i - j];
                double vec_mag = sqrt(pow(vec.x, 2) + pow(vec.y, 2));
                total_mag += vec_mag;

                //Get angle wrt x axis
                vector_angles[3-j] = acos(vec.x / vec_mag);
            }

            // //If 3 vectors are too long, skip this check. Stuck sections of path are usually short
            // if(total_mag * map_ptr->resolution > path_vertex_dist_threshold)
            //     continue;

            //Get angle rotated
            double angle_rotated = 0;
            for(int j = 0; j < vector_angles.size() - 1; ++j)
                angle_rotated += vector_angles[j] - vector_angles[j + 1];
            

            double threshold_rad = path_vertex_angle_threshold / 180.0 * M_PI;
            // if(fabs(fabs(angle_rotated) - M_PI) < path_vertex_angle_threshold/180.0 *  M_PI)
            if(fabs(angle_rotated) > M_PI - threshold_rad && fabs(angle_rotated) < M_PI + threshold_rad)
            {
                //Delete middle 2 vertices, i - 2 & i - 1, if the first and last vertices do not collide with obstacle
                if(!edgeCollides(path[i], path[i-3], collision_threshold))
                {
                    std::cout << "Vertices removed from path, points likely to be stuck. Angle rotated: " << angle_rotated << "\n";
                    i = std::distance(path.begin(), path.erase(path.begin() + i - 2, path.begin() + i)) - 1;
                }

                else
                    std::cout << "Stuck vertex found but unable to remove due to collision\n";
            }
        }
    }

    bool VoronoiPath::contractPath(std::vector<GraphNode> &path)
    {
        //Calculate minimum distance between two poses in path should have, distance in pix squared
        double waypoint_sep_sq = path_waypoint_sep * path_waypoint_sep / map_ptr->resolution / map_ptr->resolution;

        //Anchor node that is being used to check for collision
        int anchor_node = 0;

        //Anchor node that should be used for the next iteration
        int future_anchor_node = 0;

        //Used to store previous collision node for use during finding of next collision node
        int prev_collision_node = -1;

        //While anchor node has not reached the last node
        while (anchor_node < path.size() - 1)
        {
            //From anchor_node, traverse path until there is a collision, set that as the collision node
            //Trim nodes except start and end nodes
            int i;

            //Connected node is the node that anchor_node will be joined to
            int connected_node = path.size() - 1;

            //Index of node that collides with the anchor node
            int collision_node = path.size() - 1;
            for (i = prev_collision_node != -1 ? prev_collision_node : anchor_node; i < path.size(); ++i)
            {
                //If collision with node i occurs, then set the connected point as the node before i
                if (edgeCollides(path[anchor_node], path[i], trimming_collision_threshold))
                {
                    collision_node = i;
                    connected_node = i - 1;
                    break;
                }
            }

            //Remove self collision node
            if(collision_node == anchor_node)
            {
                if(anchor_node > 0)
                    anchor_node--;
                std::cout << "Self collision waypoint detected, erasing node " << collision_node << "\n";
                path.erase(path.begin() + collision_node);
                prev_collision_node = -1;
                return false;
            }

            //Preemptively set future_anchor_node, if another is found later on, this will be corrected
            future_anchor_node = collision_node;

            //Between anchor_node and connected_node, project point onto straight line connection anchor node to collision node - 1
            //Project trimmable poses on the path onto the straight line between anchor_node and connected_point
            double gradient, inv_gradient;
            try
            {
                gradient = (path[anchor_node].y - path[connected_node].y) / (path[anchor_node].x - path[connected_node].x);
                inv_gradient = -1.0 / gradient;
            }
            catch (std::exception &e)
            {
                std::cout << "Calculate gradient exception: " << e.what() << "\n";
            }

            //Project points between anchor_node and connected_node (exclusive) onto the line joining anchor_node and connected_node
            for (int j = anchor_node + 1; j < connected_node; ++j)
            {
                if (j >= path.size())
                    break;

                //Gradient is 0, shift point j's y coord to anchor_node's level
                if (gradient == 0.0)
                    path[j].y = path[anchor_node].y;

                //Gradient infinity, shit point j's x coord to anchor_node's level
                else if (fabs(gradient) == std::numeric_limits<double>::infinity())
                    path[j].x = path[anchor_node].x;

                //Valid gradient, project as normal
                else
                {
                    //Project points onto the straight line
                    //y = mx + c =====> c = y - mx
                    //(a)x + (1)y = c =====> a = -m
                    double c1 = path[connected_node].y - gradient * path[connected_node].x;
                    double c2 = path[j].y - inv_gradient * path[j].x;
                    double a1 = -gradient;
                    double a2 = -inv_gradient;
                    double determinant = a1 - a2;

                    if(determinant == 0.0)
                        std::cout << "Point projection has 0 determinant\n";

                    else
                    {
                        path[j].x = (c1 - c2) / determinant;
                        path[j].y = (a1 * c2 - a2 * c1) / determinant;
                    }
                }

                //If point is not on segment between anchor point and connected point, delete the point
                double dist = pow(path[j].x - path[j - 1].x, 2) + pow(path[j].y - path[j - 1].y, 2);

                if (!liesInSquare(path[j], path[anchor_node], path[connected_node]) || (dist < waypoint_sep_sq))
                {
                    //Decrement post deletion because the for loop will increment this again later
                    j = std::distance(path.begin(), path.erase(path.begin() + j)) - 1;

                    //Decrement connected_node, future_anchor_node, collision_node because a node before connected_node has been erased
                    --connected_node;
                    --collision_node;
                    --future_anchor_node;

                    continue;
                }

                // Also find the future anchor node, definition of future anchor node is the node that can be connected to collision node, without collision
                // If currently modified node has no collision with collision node, then it is the future anchor, break once set
                if (!edgeCollides(path[j], path[collision_node], trimming_collision_threshold))
                {
                    future_anchor_node = j;
                    break;
                }
            }

            prev_collision_node = collision_node;
            anchor_node = future_anchor_node;
        }

        return true;
    }

    void VoronoiPath::backtrack(std::vector<int> &path, double cur_dist, double last_branch_dist, const int &prev_node, const int &cur_node, std::vector<std::vector<int>> &paths, const double &backtrack_plan_threshold)
    {
        //Add cur_node to path
        path.push_back(cur_node);

        //If threshold is reached, add path to list of paths then terminate recursion
        auto prev_inf = node_inf[cur_node];
        if(prev_node != -1)
            prev_inf = node_inf[prev_node];
        auto cur_inf = node_inf[cur_node];
        double dist = sqrt(pow(prev_inf.x - cur_inf.x, 2) + pow(prev_inf.y - cur_inf.y, 2));
        cur_dist += dist;

        if(cur_dist > backtrack_plan_threshold)
        {
            paths.push_back(path);
            return;
        }

        //Each call of backtrack has its own open list of nodes to visit, which are all nodes connected to cur_node except if it already exists in the path 
        //to prevent cycles
        std::vector<int> open_list;
        for(int i = 0; i < adj_list[cur_node].size(); ++i)
        {
            if(std::find(path.begin(), path.end(), adj_list[cur_node][i]) == path.end())
                open_list.push_back(adj_list[cur_node][i]);
        }

        //If open_list is empty, it means that this node is at a dead end before reaching distance threshold, add this path to the list
        if(open_list.empty() && (cur_dist - last_branch_dist >= last_branch_dist_thresh / map_ptr->resolution))
            paths.push_back(path);

        //This node is a branch, update last branch distance with current distance
        if(open_list.size() >= 2)
            last_branch_dist = cur_dist;

        //Visit all possible paths emanating from cur_node except in direction of prev_node
        while(!open_list.empty())
        {
            backtrack(path, cur_dist, last_branch_dist, cur_node, open_list.back(), paths, backtrack_plan_threshold);

            //Remove explored node from open list
            open_list.pop_back();

            //Remove node that was added by child call
            path.pop_back();
        }
        
        return;
    }

    bool VoronoiPath::isBacktrackDistinct(std::vector<Path>::iterator &path1, std::vector<Path>::iterator &path2)
    {
        //Interpolate from path 1 endpoint to path 2 endpoint
        GraphNode start = path1->path.front();
        GraphNode vec1 = path1->path.back() - start;
        GraphNode vec2 = path2->path.back() - start;
        double mag_diff = vec2.getMagnitude() - vec1.getMagnitude();
        double angle_diff = atan2(vec2.y, vec2.x) - atan2(vec1.y, vec1.x);
        if(angle_diff > M_PI || angle_diff < -M_PI)
        {
            int sign = angle_diff / fabs(angle_diff);
            angle_diff = -sign * M_PI + angle_diff;
        }

        //TODO: Handle this properly, magnitude may be different
        // else if(fabs(angle_diff) < 0.005)
        //     return false;

        int steps = ceil(fabs(angle_diff) / 0.01);  //TODO: Hardcoded angular step size
        double mag_step = mag_diff / steps;
        double angle_step = angle_diff / steps;
        std::vector<GraphNode> interp_connection;
        GraphNode interp_vec = vec1;
        for(int step = 1; step <= steps && angle_diff > 0.005; ++step)
        {
            //Rotate interp_point vector;
            interp_vec.x = (interp_vec.x) * cos(angle_step) - (interp_vec.y) * sin(angle_step);
            interp_vec.y = (interp_vec.x) * sin(angle_step) + (interp_vec.y) * cos(angle_step);
            double mag_temp = interp_vec.getMagnitude();
            interp_vec = interp_vec / mag_temp * (mag_temp + mag_step);

            //Offset back to start position
            interp_connection.push_back(interp_vec + start);

            //Check if last 2 nodes collide with an edge, if collide means the paths are distinct. Collision threshold
            //only includes solid obstacles and not inflation zone
            if(interp_connection.size() > 1 && edgeCollides(interp_connection.back(), interp_connection[interp_connection.size() - 2], occupancy_threshold))
                return true;
        }

        //Copy and append onto path1
        interp_connection.push_back(path2->path.back());
        auto path1_copy = path1->path;
        path1_copy.insert(path1_copy.end(), interp_connection.begin(), interp_connection.end());
        path1_copy.insert(path1_copy.begin(), path2->path.front());

        //Calculate homotopy
        auto path1_homotopy = calcHomotopyClass(path1_copy);
        auto path2_homotopy = calcHomotopyClass(path2->path);

        //If homotopy is not distinct
        if(isClassDifferent(path1_homotopy, path2_homotopy))
            return true;

        else
            return false;
    }

    std::vector<std::pair<double, int>> VoronoiPath::getPathHeadings(const std::vector<Path> &paths)
    {
        std::vector<std::pair<double, int>> average_headings_pair;
        for(int i = 0; i < paths.size(); ++i)
        {
            int num_points = 0;
            double angle_sum = 0;
            double prev_angle = 0;
            //These points are in pixels, skip first point because that is start point, results in NaN angle
            for(int j = 1; j < paths[i].path.size(); ++j)
            {
                GraphNode vec(paths[i].path[j] - paths[i].path.front());
                double cur_angle = atan2(vec.y, vec.x);

                //Use angle difference to get average to prevent wrap around effect. ie a zigzag path around the 0/360 deg line will give average of 180
                double angle_diff = cur_angle - prev_angle;
                if(angle_diff > M_PI || angle_diff < -M_PI)
                {
                    int sign = angle_diff / fabs(angle_diff);
                    angle_diff = -sign * 2 * M_PI + angle_diff;
                }

                angle_sum += (prev_angle + angle_diff);
                prev_angle += angle_diff;
                num_points++;
            }

            int average_angle = angle_sum / num_points / M_PI * 180.0;
            if(average_angle < 0)
                average_angle = 360 + average_angle;

            average_headings_pair.emplace_back(average_angle, i);
        }

        return average_headings_pair;
    }

    std::vector<Path> VoronoiPath::backtrackPlan(const GraphNode &start)
    {
        //Block until voronoi is no longer being updated. Prevents issue where planning is done using an empty adjacency list
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        Profiler complete_profiler, section_profiler;
        
        //Compensate previous path's map origin
        if(prev_map.frame_id != "")
        {
            backtrack_paths = true;
            double prev_origin_x = prev_map.origin.position.x / prev_map.resolution;
            double prev_origin_y = prev_map.origin.position.y / prev_map.resolution;
            double cur_origin_x = map_ptr->origin.position.x / map_ptr->resolution;
            double cur_origin_y = map_ptr->origin.position.y / map_ptr->resolution;

            for(auto & path: previous_paths)
            {
                for(auto &pose : path.path)
                {
                    pose.x = pose.x + prev_origin_x - cur_origin_x;
                    pose.y = pose.y + prev_origin_y - cur_origin_y;
                }
            }
        }

        if(print_timings)
            section_profiler.print("backtrackPlan offset previous timestep path");

        //Update previous map information
        prev_map.frame_id = map_ptr->frame_id;
        prev_map.origin.position.x = map_ptr->origin.position.x;
        prev_map.origin.position.y = map_ptr->origin.position.y;
        prev_map.resolution = map_ptr->resolution;

        std::vector<std::vector<int>> paths;

        //Get nearest node from robot's position
        //Find nearest node to starting and end positions
        int start_node;
        if (!getNearestNode(start, start, start_node, start_node))
            return std::vector<Path>();

        if(print_timings)
            section_profiler.print("backtrackPlan getNearestNode");

        //Run exhaustive traversal of connected nodes until termination condition is met
        //Termination condition is distance threshold reached
        std::vector<int> path;
        backtrack(path, 0, 0, -1, start_node, paths, backtrack_plan_threshold / map_ptr->resolution);

        if(print_timings)
            section_profiler.print("backtrackPlan recursive backtrack");

        //Convert to pixel paths
        std::vector<Path> all_path_nodes;
        for (int i = 0; i < paths.size(); ++i)
        {
            all_path_nodes.emplace_back(getUniqueID(), std::vector<GraphNode>{start});
            all_path_nodes[i].path.reserve(paths[i].size() + 2);

            for (const auto &node : paths[i])
                all_path_nodes[i].path.emplace_back(node_inf[node].x, node_inf[node].y);
        }

        interpolateContractPaths(all_path_nodes);

        if(print_timings)
            section_profiler.print("backtrackPlan interpolate and contract");

        std::vector<int> remove_ind;
        auto prev_path_headings = getPathHeadings(previous_paths);
        do{
            //Clear from previous iteration
            remove_ind.clear();

            //Sort paths by average heading
            std::vector<std::pair<double, int>> average_headings_pair = getPathHeadings(all_path_nodes);

            std::sort(average_headings_pair.begin(), average_headings_pair.end());
            std::vector<Path> sorted_paths;
            std::vector<double> sorted_average_headings;
            for(int j = 0; j < average_headings_pair.size(); ++j)
            {
                sorted_paths.emplace_back(std::move(all_path_nodes[average_headings_pair[j].second]));
                sorted_average_headings.push_back(average_headings_pair[j].first);
            }
            all_path_nodes = std::move(sorted_paths);

            //Check all paths with their adjacent paths
            for(auto i = all_path_nodes.begin(); i < all_path_nodes.end(); ++i)
            {
                auto path1 = i;
                auto path2 = i + 1;
                if(path2 == all_path_nodes.end())
                {
                    if(all_path_nodes.size() > 2)
                        path2 = all_path_nodes.begin();

                    else 
                        break;
                }

                //Check if heading exceeds threshold
                bool distinct = false;
                int path1_ind = std::distance(all_path_nodes.begin(), path1);
                int path2_ind = std::distance(all_path_nodes.begin(), path2);
                double heading_diff = sorted_average_headings[path1_ind] - sorted_average_headings[path2_ind];
                if(heading_diff > 180 || heading_diff < -180)
                {
                    int sign = -heading_diff / fabs(heading_diff);
                    heading_diff = sign * 360 + heading_diff;
                }

                //If headings greater than threshold, then distinct, otherwise check using interpolation and homotopy
                if(fabs(heading_diff) < 30 && !isBacktrackDistinct(path1, path2))
                {
                    //Path removal should favor the path that has close relative in previous time step to prevent oscillation
                    std::vector<Path> candidate_paths;
                    candidate_paths.push_back(*path1);
                    candidate_paths.push_back(*path2);
                    auto candidate_path_headings = getPathHeadings(candidate_paths);

                    std::vector<double> mins;
                    mins.resize(candidate_path_headings.size());
                    for(int p = 0; p < candidate_path_headings.size(); ++p)
                    {
                        double min = 1000000;
                        for(int l = 0; l < prev_path_headings.size(); ++l)
                        {
                            double angle_diff = fabs(prev_path_headings[l].first - candidate_path_headings[p].first);
                            if(angle_diff < min)
                                min = angle_diff;
                        }

                        mins[p] = min;
                    }

                    //If both are greater than 45deg, use distance instead
                    if(mins[0] < 45 || mins[1] < 45)
                    {
                        //Path 1 has a nearer relative, erase path 2
                        if(mins[0] < mins[1])
                            remove_ind.push_back(std::distance(all_path_nodes.begin(), path2));

                        else
                            remove_ind.push_back(std::distance(all_path_nodes.begin(), path1));
                    }

                    else
                    {
                        //Remove path that has endpoint nearer to robot position
                        double dist1 = sqrt(pow(start.x - path1->path.back().x, 2) + pow(start.y - path1->path.back().y, 2));
                        double dist2 = sqrt(pow(start.x - path2->path.back().x, 2) + pow(start.y - path2->path.back().y, 2));

                        //Erasing returns iterator of object AFTER erased object, -1 because for loop will increment again
                        if(dist1 < dist2)
                            remove_ind.push_back(std::distance(all_path_nodes.begin(), path1));

                        else
                            remove_ind.push_back(std::distance(all_path_nodes.begin(), path2));
                    }
                }
            }

            //Copy non-removed paths only
            std::vector<Path> copy;
            for(int i = 0; i < all_path_nodes.size(); ++i)
            {
                //This index is not to be removed
                if(std::find(remove_ind.begin(), remove_ind.end(), i) == remove_ind.end())
                    copy.emplace_back(std::move(all_path_nodes[i]));
            }
            all_path_nodes = std::move(copy);

        } while(!remove_ind.empty());

        if(print_timings)
            section_profiler.print("backtrackPlan check unique");

        //If there were no previous paths, then just accept these paths and return
        if(!hasPreviousPaths())
            previous_paths = all_path_nodes;

        //Otherwise, try to link previous and current paths 
        else
        {
            viz_paths = linkBacktrackPaths(previous_paths, all_path_nodes);
            previous_paths = all_path_nodes;
        }

        if(print_timings)
            complete_profiler.print("backtrackPlan total time");
        return all_path_nodes;
    }

    std::vector<Path> VoronoiPath::getVizPaths()
    {
        return viz_paths;
    }

    std::vector<Path> VoronoiPath::linkBacktrackPaths(std::vector<Path> &prev_paths, std::vector<Path> &cur_paths)
    {
        //Compare paths within a certain heading +- range from current path's heading
        Profiler complete_profiler;
        auto prev_path_headings = getPathHeadings(prev_paths);
        auto cur_path_headings = getPathHeadings(cur_paths);
        std::vector<Path> viz_paths = cur_paths;

        //<int, double> = <path index, average heading>
        for(int i = 0; i < cur_path_headings.size(); ++i)
        {
            auto cur_path = cur_paths.begin() + cur_path_headings[i].second;
            double cur_heading = cur_path_headings[i].first;

            //Duplicate prev_path_headings and then offset by cur_path, then sort. Compare smallest angle ones first
            auto prev_headings_copy = prev_path_headings;
            for(auto &heading : prev_headings_copy)
            {
                double old_heading = heading.first;
                heading.first -= cur_heading;
                if(heading.first > 180 || heading.first < -180)
                {
                    int sign = -heading.first/fabs(heading.first);
                    heading.first = sign * 360 + heading.first;
                }

                heading.first = fabs(heading.first);
            }
            std::sort(prev_headings_copy.begin(), prev_headings_copy.end());
            
            //Compare starting from smallest absolute angle difference, increase likelihood of comparing with actual predecessor
            bool distinct = true;
            for(int j = 0; j < prev_headings_copy.size(); ++j)
            {
                auto prev_path = prev_paths.begin() + prev_headings_copy[j].second;
                double heading_diff = prev_headings_copy[j].first;

                if(fabs(heading_diff) < 30)
                {
                    //Check if prev_path collides with obstacles in this map frame
                    bool collides = false;
                    for(int k = 1; k < prev_path->path.size(); ++k)
                    {
                        if(edgeCollides(prev_path->path[k-1], prev_path->path[k], occupancy_threshold))
                        {
                            std::cout << "Path " << i << " collides\n";
                            collides = true;
                            break;
                        }
                    }

                    //Not distinct, assign current id to previous id if previous id is still valid
                    if(!collides && prev_path->id != 0 && !isBacktrackDistinct(cur_path, prev_path))
                    {
                        cur_path->id = prev_path->id;
                        prev_path->id = 0;
                        distinct = false;
                        
                        //For visualization
                        viz_paths[i].path.insert(viz_paths[i].path.end(), prev_path->path.rbegin(), prev_path->path.rend());

                        //Only do 1 comparison per path
                        //TODO: Robustify this
                        break;
                    }
                }
            }

            if(distinct)
                std::cout << "Path " << i << " is distinct\n";
        }

        if(print_timings)
            complete_profiler.print("linkBacktrackPaths total time");

        return viz_paths;
    }


    std::vector<Path> VoronoiPath::getPath(const GraphNode &start, const GraphNode &end, const int &num_paths)
    {
        //Block until voronoi is no longer being updated. Prevents issue where planning is done using an empty adjacency list
        std::lock_guard<std::mutex> lock(voronoi_mtx);

        Profiler complete_profiler, section_profiler;
        std::vector<Path> path;
        backtrack_paths = false;

        //Find nearest node to starting and end positions
        int start_node, end_node;
        if (!getNearestNode(start, end, start_node, end_node))
            return std::vector<Path>();

        if (print_timings)
            section_profiler.print("getPath find nearest node");

        std::vector<int> shortest_path;
        if (findShortestPath(start_node, end_node, shortest_path))
        {
            if (print_timings)
                section_profiler.print("getPath find shortest path");

            std::vector<std::vector<int>> all_paths;
            //Get next shortest path
            if (num_paths >= 1)
                kthShortestPaths(start_node, end_node, shortest_path, all_paths, num_paths - 1);

            if (print_timings)
                section_profiler.print("getPath find kth shortest paths");

            //Copy all_paths into new container which include start and end
            std::vector<Path> all_path_nodes;
            all_path_nodes.reserve(all_paths.size());
            for (int i = 0; i < all_paths.size(); ++i)
            {
                all_path_nodes.emplace_back(getUniqueID(), std::vector<GraphNode>{start});
                all_path_nodes[i].path.reserve(all_paths[i].size() + 2);

                for (const auto &node : all_paths[i])
                    all_path_nodes[i].path.emplace_back(node_inf[node].x, node_inf[node].y);

                all_path_nodes[i].path.push_back(end);
            }

            if (print_timings)
                section_profiler.print("getPath insert start and end");

            //Trim beginning of path to remove unnecessary u-turns in path
            interpolateContractPaths(all_path_nodes);

            if (print_timings)
                section_profiler.print("getPath interpolate and contract");

            //Only set previous paths and their costs if this was the first getPath call
            if (!hasPreviousPaths())
            {
                previous_paths = all_path_nodes;
                previous_path_costs = std::vector<double>(all_path_nodes.size(), 0);

                for (int j = 0; j < all_path_nodes.size(); ++j)
                {
                    for (int i = 0; i < all_path_nodes[j].path.size() - 1; ++i)
                        previous_path_costs[j] += euclideanDist(all_path_nodes[j].path[i], all_path_nodes[j].path[i + 1]);
                }

                //Swap minimum cost path with first in list, sometimes after contraction the first index path is no longer the shortest
                auto min_it = std::min_element(previous_path_costs.begin(), previous_path_costs.end());
                int ind = std::distance(previous_path_costs.begin(), min_it);
                std::swap(previous_path_costs[0], previous_path_costs[ind]);
                std::swap(all_path_nodes[0], all_path_nodes[ind]);

                if (print_timings)
                    section_profiler.print("getPath get all initial costs");
            }

            path = std::move(all_path_nodes);

            if (print_timings)
                complete_profiler.print("getPath find all paths");
        }

        else
            std::cout << "Path could not be found" << std::endl;

        return path;
    }

    std::vector<Path> VoronoiPath::replan(GraphNode &start, GraphNode &end, int num_paths, int &pref_path)
    {
        Profiler complete_profiler, contract_profiler;
        if (previous_paths.empty())
            return previous_paths;

        /********** TRIMMING OR EXTENSION OF PATHS FOUND IN PREVIOUS TIME STEP **********/
        //Add robot's current position to the first pose of the replanned_paths
        bool found_new_start = false;
        backtrack_paths = false;
        std::vector<Path> replanned_paths(previous_paths);
        for (int i = 0; i < replanned_paths.size(); ++i)
        {
            //Search nearby area around robot to find an empty cell to connect to the previous path
            if (!found_new_start && edgeCollides(replanned_paths[i].path[0], start, trimming_collision_threshold))
            {
                //Current radius in pixels and angle in rads
                double current_radius = 1.0, current_angle = 0;
                double max_pix_radius = search_radius / map_ptr->resolution;
                while (current_radius <= max_pix_radius && !found_new_start)
                {
                    current_angle = 0;
                    double increment = 1.0 / current_radius; // s = r(theta), solve theta such that s == 1, 1 pixel
                    while (current_angle < 2 * M_PI && !found_new_start)
                    {
                        // If the candidate start position doesn't collide with anything when joined to first pose of path
                        GraphNode candidate_start(start.x + cos(current_angle) * current_radius, start.y + sin(current_angle) * current_radius);
                        if (!edgeCollides(candidate_start, replanned_paths[i].path[0], trimming_collision_threshold))
                        {
                            start = candidate_start;
                            found_new_start = true;
                        }

                        current_angle += increment;
                    }

                    current_radius += 1.0;
                }

                //No nearby empty cell found
                if (!found_new_start)
                    std::cout << "WARN: No empty cell within search radius: " << search_radius << "m\n";
            }

            //Insert the robot's current position/new starting posititon as the first pose of path
            replanned_paths[i].path.insert(replanned_paths[i].path.begin(), start);
        }

        interpolateContractPaths(replanned_paths);

        if (print_timings)
            contract_profiler.print("replan contract and join total");

        /********** HOMOTOPY EXPLORATION TO FIND NEW PATHS **********/
        //Explore for potential paths in new homotopy classes
        std::vector<Path> potential_paths = getPath(start, end, num_paths / 2);

        //Calculate homotopy class of previous set of paths
        Profiler homotopy_profiler;
        std::vector<std::complex<double>> previous_classes;
        for (const auto &path : replanned_paths)
            previous_classes.push_back(calcHomotopyClass(path.path));

        if (print_timings)
            homotopy_profiler.print("replan homotopy calc");

        // Add potential paths that are unique to replanned_paths container
        for (const auto &path : potential_paths)
        {
            std::complex<double> temp_class = calcHomotopyClass(path.path);
            bool unique = true;
            for (const auto &prev_class : previous_classes)
                if (!isClassDifferent(temp_class, prev_class))
                    unique = false;

            if (unique)
                replanned_paths.push_back(path);
        }

        if (print_timings)
            homotopy_profiler.print("replan check homotopy unique");

        //Get cost of all the paths
        std::vector<double> all_paths_cost(replanned_paths.size(), 0);
        for (int j = 0; j < replanned_paths.size(); ++j)
        {
            for (int i = 0; i < replanned_paths[j].path.size() - 1; ++i)
                all_paths_cost[j] += euclideanDist(replanned_paths[j].path[i], replanned_paths[j].path[i + 1]);
        }

        if (print_timings)
            homotopy_profiler.print("replan get all costs");

        //Assign -ve infinity cost to currently (selected) preferred path, prevent deletion
        double pref_path_cost = all_paths_cost[pref_path];
        all_paths_cost[pref_path] = -std::numeric_limits<double>::infinity();

        //If number of paths greater than num_paths, delete longest paths until equal
        while (replanned_paths.size() > num_paths)
        {
            auto max_it = std::max_element(all_paths_cost.begin(), all_paths_cost.end());
            int ind = std::distance(all_paths_cost.begin(), max_it);

            //Order of paths < num_paths must be maintained
            //If max element is greater than or equal to num_paths limit, means it is a potential path, just delete
            if (ind >= num_paths)
            {
                all_paths_cost.erase(max_it);
                replanned_paths.erase(replanned_paths.begin() + ind);
            }

            //Path is < num_paths, swap with one of the paths behind and then delete last
            else
            {
                std::swap(replanned_paths[ind], replanned_paths[replanned_paths.size() - 1]);
                std::swap(all_paths_cost[ind], all_paths_cost[all_paths_cost.size() - 1]);

                replanned_paths.pop_back();
                all_paths_cost.pop_back();
            }
        }

        //Restore pref path cost
        all_paths_cost[pref_path] = pref_path_cost;

        //Update previous paths and their costs for the next round of replanning
        previous_paths = replanned_paths;
        previous_path_costs = all_paths_cost;

        if (print_timings)
            complete_profiler.print("replan total replan time");

        return replanned_paths;
    }

    bool VoronoiPath::getNearestNode(const GraphNode &start, const GraphNode &end, int &start_node, int &end_node)
    {
        bool find_end = true;
        if(start == end)
            find_end = false;

        //TODO: Should not only check nearest nodes. Should allow nearest position to be on an edge
        double min_start_dist = std::numeric_limits<double>::infinity();
        double min_end_dist = std::numeric_limits<double>::infinity();
        start_node = -1;
        end_node = -1;

        //Traverse all nodes to find the one with minimum distance from start and end points
        for (int i = 0; i < num_nodes; ++i)
        {
            if (adj_list[i].empty())
                continue;

            GraphNode curr = node_inf[i];
            double temp_start_dist = pow(curr.x - start.x, 2) + pow(curr.y - start.y, 2);
            if (temp_start_dist < min_start_dist)
            {
                if (!edgeCollides(start, curr, collision_threshold))
                {
                    min_start_dist = temp_start_dist;
                    start_node = i;
                }
            }

            if(find_end)
            {
                double temp_end_dist = pow(curr.x - end.x, 2) + pow(curr.y - end.y, 2);
                if (temp_end_dist < min_end_dist)
                {
                    if (!edgeCollides(end, curr, collision_threshold))
                    {
                        min_end_dist = temp_end_dist;
                        end_node = i;
                    }
                }
            }
        }

        //Failed to find start/end even after relaxation
        if (start_node == -1 || (find_end && end_node == -1))
        {
            std::cout << "Failed to find nearest starting or ending node" << std::endl;
            return false;
        }

        return true;
    }

    std::vector<GraphNode> VoronoiPath::convertToPixelPath(const std::vector<int> &path_)
    {
        std::vector<GraphNode> return_path;
        return_path.reserve(path_.size());
        for (const auto &node : path_)
            return_path.emplace_back(node_inf[node].x, node_inf[node].y);

        return return_path;
    }

    //https://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-216.pdf
    std::complex<double> VoronoiPath::calcHomotopyClass(const std::vector<GraphNode> &path_)
    {
        std::vector<std::complex<double>> path;
        path.reserve(path_.size());

        //Convert path to complex path
        for (const auto &node : path_)
            path.emplace_back(node.x, node.y);

        //Go through each edge of the path and calculate its homotopy value
        std::complex<double> path_sum(0, 0);
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<std::complex<double>>> future_vector;
        future_vector.reserve(num_threads);

        int poses_per_thread = path.size() / num_threads;

        for (int i = 0; i < num_threads; ++i)
        {
            int start_pose = i * poses_per_thread;

            //Last thread takes remaining poses
            if (i == num_threads - 1)
                poses_per_thread = path.size() - poses_per_thread * (num_threads - 1);

            future_vector.emplace_back(std::async(
                std::launch::async,
                [&, start_pose, poses_per_thread, path]() {
                    std::complex<double> path_sum(0, 0);
                    for (int i = start_pose + 1; i < start_pose + poses_per_thread + 1; i++)
                    {
                        std::complex<double> edge_sum(0, 0);
                        if (i >= path.size())
                            continue;

                        //Each edge must iterate through all obstacles
                        for (int j = 0; j < centers.size(); ++j)
                        {
                            double real_part = std::log(std::abs(path[i] - centers[j])) - std::log(std::abs(path[i - 1] - centers[j]));
                            double im_part = std::arg(path[i] - centers[j]) - std::arg(path[i - 1] - centers[j]);

                            //Get smallest angle
                            while (im_part > M_PI)
                                im_part -= 2 * M_PI;

                            while (im_part < -M_PI)
                                im_part += 2 * M_PI;

                            edge_sum += (std::complex<double>(real_part, im_part) * obs_coeff[j]);
                        }
                        //Add this edge's sum to the path sum
                        path_sum += edge_sum;
                    }

                    return path_sum;
                }));
        }

        for (int i = 0; i < future_vector.size(); ++i)
        {
            std::complex<double> temp_sum = future_vector[i].get();
            path_sum += temp_sum;
        }

        return path_sum;
    }

    bool VoronoiPath::kthShortestPaths(const int &start_node, const int &end_node, const std::vector<int> &shortestPath, std::vector<std::vector<int>> &all_paths, const int &num_paths)
    {
        //Reserve num_paths + 1, + 1 is to store the original shortest path
        all_paths.reserve(num_paths + 1);

        if (num_paths == 0)
        {
            all_paths.push_back(shortestPath);
            return true;
        }

        //Create container to store accepted kth shortest paths
        std::vector<std::vector<int>> kthPaths;
        kthPaths.reserve(num_paths + 1);
        kthPaths.push_back(shortestPath);

        //Create backup of adjacency as well as modified indices to reduce time required for restoring original list
        std::vector<std::vector<int>> adj_list_backup(adj_list);
        std::vector<int> adj_list_modified_ind;

        //Containers to store candidate kth shortest paths and their costs, and homotopy classes of all accepted paths
        std::vector<std::pair<double, std::vector<int>>> potentialKth;
        std::vector<std::complex<double>> homotopy_classes;

        for (int k = 1; k <= num_paths; ++k)
        {
            //Break if cannot find the previous k path, could not find all NUM_PATHS
            if (k - 1 == kthPaths.size())
                break;

            //Update homotopy classes vector whenever new kth path gets added
            homotopy_classes.push_back(calcHomotopyClass(convertToPixelPath(kthPaths.back())));

            //Spur node is ith node, from start to 2nd last node of path, inclusive
            for (int i = 0; i < kthPaths[k - 1].size() - 1; ++i)
            {
                int spurNode = kthPaths[k - 1][i];

                //Copy root path into container. Root path is path up until spur node, containing the path from start onwards
                std::vector<int> rootPath(i + 1);
                std::copy(kthPaths[k - 1].begin(), kthPaths[k - 1].begin() + i + 1, rootPath.begin());

                //Disconnect edges if root path has already been discovered before
                for (const auto &prevKthPath : kthPaths)
                {
                    int equal_count = 0;
                    for (int z = 0; z <= i; ++z)
                    {
                        if (z >= prevKthPath.size() || z >= rootPath.size())
                            break;

                        if (prevKthPath[z] == rootPath[z])
                            equal_count++;
                    }

                    //Entire root path is identical to previously discovered kth path, disconnect edge between spur_next and spurNode
                    if (equal_count == rootPath.size() && i + 1 < prevKthPath.size())
                    {
                        //Remove edge from spurNode to spur_next by setting spur_next in spurNode's adjacency to -1
                        int spur_next = prevKthPath[i + 1];
                        auto erase_it = std::find(adj_list[spurNode].begin(), adj_list[spurNode].end(), spur_next);
                        if (erase_it != adj_list[spurNode].end())
                        {
                            adj_list[spurNode][erase_it - adj_list[spurNode].begin()] = -1;
                            adj_list_modified_ind.push_back(spurNode);
                        }

                        //Remove edge from spur_next to spurNode by setting spurNode in spur_next's adjacency to -1
                        erase_it = std::find(adj_list[spur_next].begin(), adj_list[spur_next].end(), spurNode);
                        if (erase_it != adj_list[spur_next].end())
                        {
                            adj_list[spur_next][erase_it - adj_list[spur_next].begin()] = -1;
                            adj_list_modified_ind.push_back(spur_next);
                        }
                    }
                }

                //Remove all nodes of root path from graph except spur node and start node
                //Exclude spurNode (rootPath.back())
                for (int node_ind = 0; node_ind < rootPath.size() - 1; ++node_ind)
                {
                    int node = rootPath[node_ind];
                    for (int del_ind = 0; del_ind < adj_list[node].size(); ++del_ind)
                    {
                        //Remove connection from point-er side
                        int pointed_to = adj_list[node][del_ind];

                        //Edge is not deleted yet, continue to delete
                        if (pointed_to != -1)
                        {
                            adj_list[node][del_ind] = -1;
                            adj_list_modified_ind.push_back(node);

                            //Remove connection from point-ed side
                            auto erase_it = std::find(adj_list[pointed_to].begin(), adj_list[pointed_to].end(), node);
                            if (erase_it != adj_list[pointed_to].end())
                            {
                                adj_list[pointed_to][erase_it - adj_list[pointed_to].begin()] = -1;
                                adj_list_modified_ind.push_back(pointed_to);
                            }
                        }
                    }
                }

                //Find spur path starting from spur node using A* algorithm shortest path searching with modified adj_list
                std::vector<int> spur_path;
                if (findShortestPath(spurNode, end_node, spur_path))
                {
                    //Create full path from root path and spur path
                    std::vector<int> total_path;
                    if (rootPath.size())
                        total_path.insert(total_path.begin(), rootPath.begin(), rootPath.end() - 1);
                    total_path.insert(total_path.end(), spur_path.begin(), spur_path.end());

                    //Make sure to check all paths, one container might be bigger. Set last ind to larger container's size
                    int last = kthPaths.size() > potentialKth.size() ? kthPaths.size() : potentialKth.size();

                    //Check if the path just generated is unique in the potential kth paths
                    bool path_is_unique = true;
                    for (int check_path = 0; check_path < last; ++check_path)
                    {
                        int kth_equal = 0;
                        int pot_equal = 0;

                        for (const auto &node_pot : total_path)
                        {
                            //Compare with kthPaths
                            if (check_path < kthPaths.size())
                            {
                                if (node_pot < kthPaths[check_path].size())
                                    if (kthPaths[check_path][node_pot] == total_path[node_pot])
                                        kth_equal++;
                            }

                            //Compare with potentialKths
                            if (check_path < potentialKth.size())
                            {
                                if (node_pot < potentialKth[check_path].second.size())
                                    if (potentialKth[check_path].second[node_pot] == total_path[node_pot])
                                        pot_equal++;
                            }
                        }

                        if (pot_equal == total_path.size() || kth_equal == total_path.size())
                            path_is_unique = false;
                    }

                    //Add unique path to list of potential kth paths
                    if (path_is_unique)
                    {
                        //Get cost of total path
                        double total_cost = 0;
                        for (int int_node = 0; int_node < total_path.size() - 1; ++int_node)
                            total_cost += euclideanDist(node_inf[total_path[int_node]], node_inf[total_path[int_node + 1]]);

                        //Store path and its corresponding cost as a pair
                        // cost_index_vec.emplace_back(total_cost, potentialKth.size());
                        potentialKth.emplace_back(total_cost, std::move(total_path));
                    }
                }

                //Reset adj_list before changing spur node
                for (const auto &modified_node : adj_list_modified_ind)
                    adj_list[modified_node] = adj_list_backup[modified_node];

                adj_list_modified_ind.clear();
                adj_list_modified_ind.shrink_to_fit();
            }

            //No alternate paths found
            if (potentialKth.size() == 0)
                break;

            //Sort costs of paths
            std::sort(potentialKth.begin(), potentialKth.end());

            //Check whether paths are unique and erase non-unique paths, starting from lowest cost path, breaks when lowest cost path is unique
            auto path_it = potentialKth.begin();
            int h = 0; //Path is unique if h == homotopy_classes.size()
            while (path_it < potentialKth.end() && h != homotopy_classes.size())
            {
                //Get homotopy class of the path that is currently being considered
                std::complex<double> curr_h_class = calcHomotopyClass(convertToPixelPath(path_it->second));
                for (h = 0; h < homotopy_classes.size(); ++h)
                {
                    //Path is not unique
                    if (!isClassDifferent(curr_h_class, homotopy_classes[h]))
                    {
                        //Erase and then break if not unique, set path_it to the new iterator of path behind deleted path
                        path_it = potentialKth.erase(path_it);
                        break;
                    }
                }
            }

            //If there are remaining unique paths, add the shortest one to kthPaths
            if (!potentialKth.empty())
            {
                kthPaths.push_back(std::move(potentialKth[0].second));
                potentialKth.erase(potentialKth.begin());
            }
        }

        all_paths.insert(all_paths.begin(), kthPaths.begin(), kthPaths.end());
        if (num_paths == all_paths.size() - 1)
            return true;

        else
            return false;
    }

    bool VoronoiPath::findShortestPath(const int &start_node, const int &end_node, std::vector<int> &path)
    {
        //Create open list, boolean closed_list, and list storing previous node required to reach node at index i
        std::vector<std::pair<int, NodeInfo>> open_list;
        std::vector<bool> nodes_closed_bool(num_nodes, false);
        std::vector<int> nodes_prev(num_nodes, -1);

        //Variable to store starting node's A* parameters
        NodeInfo start_info;
        start_info.cost_upto_here = 0;
        start_info.cost_to_goal = euclideanDist(node_inf[start_node], node_inf[end_node]);
        start_info.updateCost();

        //Place first node into open list to begin exploration
        open_list.emplace_back(std::make_pair(start_node, start_info));

        GraphNode end_node_location = node_inf[end_node];
        GraphNode next_node_location;
        NodeInfo curr_node_info;
        int next_node;

        //Run until the end_node enters the closed list
        while (!nodes_closed_bool[end_node])
        {
            //Get current node from first item of open list
            int curr_node = open_list[0].first;
            GraphNode curr_node_location = node_inf[curr_node];

            //Get info for current node
            curr_node_info = open_list[0].second;

            //Loop all adjacent nodes of current node
            for (int i = 0; i < adj_list[curr_node].size(); ++i)
            {
                next_node = adj_list[curr_node][i];

                //Edge has been deleted or node is already in closed list
                if (next_node == -1 || nodes_closed_bool[next_node])
                    continue;

                //Get the location of the next node
                next_node_location = node_inf[next_node];

                //Calculate cost upto the next node from curr node
                double curr_to_next_dist = euclideanDist(curr_node_location, next_node_location) + curr_node_info.cost_upto_here;

                //Find next_node in open_list to check whether or not to add into open list
                auto it = std::find_if(open_list.begin(), open_list.end(),
                                       [&next_node](const std::pair<int, NodeInfo> &in) {
                                           return in.first == next_node;
                                       });

                //If node is not in open list yet, add to open list
                if (it == open_list.end())
                {
                    NodeInfo new_node;
                    new_node.cost_upto_here = curr_to_next_dist;
                    new_node.cost_to_goal = euclideanDist(end_node_location, next_node_location);
                    new_node.updateCost();
                    nodes_prev[next_node] = curr_node;

                    open_list.emplace_back(std::make_pair(next_node, new_node));
                }

                //Update open list's cost if this new cost is smaller
                else
                {
                    //Update node's nearest distance to reach here, if the new cost is lower
                    if (curr_to_next_dist < it->second.cost_upto_here)
                    {
                        //Update cost upto here. Cost to goal doesn't change
                        it->second.cost_upto_here = curr_to_next_dist;
                        it->second.updateCost();
                        nodes_prev[next_node] = curr_node;
                    }
                }
            }

            //Remove curr_node from open list after all adjacent nodes have been put into open list
            open_list.erase(open_list.begin());

            //Then put into closed list
            nodes_closed_bool[curr_node] = true;

            //Sort open list by ascending total cost
            if (!open_list.empty())
            {
                //Sort by pair's second element.total_cost
                std::sort(open_list.begin(), open_list.end(), [](std::pair<int, NodeInfo> &left, std::pair<int, NodeInfo> &right) {
                    return left.second.total_cost < right.second.total_cost;
                });
            }

            //No path is found since end node is not in closed list and open list is empty
            else
                return false;
        }

        //Put nodes into path
        std::vector<int> temp_path;

        //Find path starting from end_node, using nodes_prev to backtrack path
        int path_current_node = end_node;
        temp_path.push_back(path_current_node);

        //Loop until start_node has been reached
        while (path_current_node != start_node)
        {
            path_current_node = nodes_prev[path_current_node];

            //If previous node does not exist, dead end. Path does not exist
            if (path_current_node == -1)
                return false;

            temp_path.push_back(path_current_node);
        }

        //Insert path found in reverse order
        path.insert(path.begin(), temp_path.rbegin(), temp_path.rend());

        return true;
    }

    void VoronoiPath::removeObstacleVertices(std::vector<const jcv_edge *> &edge_vector)
    {
        //Get edge vertices that are in obtacle
        //Data loaded by map server is upside down. Top of image is last of data array
        //Left right order is the same as in image
        //Meaning map.data reads from image from bottom of image, upwards, left to right
        std::vector<int> delete_indices;
        for (int i = 0; i < edge_vector.size(); ++i)
        {
            //Check each vertex if is inside obstacle
            for (int j = 0; j < 2; ++j)
            {
                int pixel = floor(edge_vector[i]->pos[j].x) + floor(edge_vector[i]->pos[j].y) * map_ptr->width;

                //If vertex pixel in map is not free, remove this edge
                if (map_ptr->data[pixel] > collision_threshold)
                {
                    delete_indices.push_back(i);
                    break;
                }
            }
        }

        if (delete_indices.size() != 0)
        {
            std::vector<const jcv_edge *> remaining_edges;
            int delete_count = 0;
            for (int i = 0; i < edge_vector.size(); ++i)
            {
                if (delete_indices[delete_count] == i)
                {
                    delete_count++;
                    continue;
                }

                remaining_edges.push_back(edge_vector[i]);
            }

            //Copy remaining edges into original container
            edge_vector.clear();
            edge_vector.insert(edge_vector.begin(), remaining_edges.begin(), remaining_edges.end());
        }

        // //This method is slower
        // auto it = edge_vector.begin();
        // while (it < edge_vector.end())
        // {
        //     //Check each vertex if is inside obstacle
        //     for (int j = 0; j < 2; ++j)
        //     {
        //         int pixel = floor(it->pos[j].x) + floor(it->pos[j].y) * map.width;

        //         //If vertex pixel in map is not free, remove this edge
        //         if (map.data[pixel] > collision_threshold)
        //         {
        //             it = edge_vector.erase(it);
        //             continue;
        //         }
        //     }

        //     ++it;
        // }
    }

    void VoronoiPath::removeCollisionEdges(std::vector<const jcv_edge *> &edge_vector)
    {
        std::vector<int> delete_indices;
        for (int i = 0; i < edge_vector.size(); ++i)
        {
            const jcv_edge *curr_edge = edge_vector[i];

            GraphNode start(curr_edge->pos[0].x, curr_edge->pos[0].y);
            GraphNode end(curr_edge->pos[1].x, curr_edge->pos[1].y);

            if (edgeCollides(start, end, collision_threshold))
                delete_indices.push_back(i);
        }

        if (delete_indices.size() != 0)
        {
            std::vector<const jcv_edge *> remaining_edges;
            int delete_count = 0;
            for (int i = 0; i < edge_vector.size(); ++i)
            {
                if (delete_indices[delete_count] == i)
                {
                    delete_count++;
                    continue;
                }

                remaining_edges.push_back(edge_vector[i]);
            }

            //Copy remaining edges into original container
            edge_vector.clear();
            edge_vector.insert(edge_vector.begin(), remaining_edges.begin(), remaining_edges.end());
        }

        // //This method is slower
        // auto it = edge_vector.begin();
        // while (it != edge_vector.end())
        // {
        //     jcv_edge curr_edge = *it;

        //     GraphNode start(curr_edge.pos[0].x, curr_edge.pos[0].y);
        //     GraphNode end(curr_edge.pos[1].x, curr_edge.pos[1].y);

        //     //If vertex pixel in map is not free, remove this edge
        //     if (edgeCollides(start, end))
        //     {
        //         it = edge_vector.erase(it);
        //         continue;
        //     }

        //     ++it;
        // }
    }

    double VoronoiPath::vectorAngle(const double vec1[2], const double vec2[2])
    {
        double dot = vec1[0] * vec2[0] + vec1[1] * vec2[1];
        double det = vec1[0] * vec2[1] - vec1[1] * vec2[0];
        return std::atan2(det, dot);
    }

    bool VoronoiPath::edgeCollides(const GraphNode &start, const GraphNode &end, int threshold)
    {
        //Check start and end cells first
        try
        {
            if (map_ptr->data.at(floor(start.x) + floor(start.y) * map_ptr->width) > threshold)
                return true;

        }
        catch (const std::exception &e)
        {
            std::cout << "Exception occurred with edge collision checking on start node, " << e.what() << std::endl;
        }
        try
        {
            if (map_ptr->data.at(floor(end.x) + floor(end.y) * map_ptr->width) > threshold)
                return true;
        }
        catch (const std::exception &e)
        {
            std::cout << "Exception occurred with edge collision checking on end node, " << e.what() << std::endl;
        }

        double steps = 0;
        double distance = sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2));

        if (distance > line_check_resolution)
            steps = distance / line_check_resolution;

        else
            steps = 1;

        //Calculate linear increment in x and y to reduce computation costs
        double increment_x = (end.x - start.x) / (double)steps;
        double increment_y = (end.y - start.y) / (double)steps;
        GraphNode curr_node = start;

        for (int i = 0; i <= steps; ++i)
        {
            int pixel = floor(curr_node.x) + floor(curr_node.y) * map_ptr->width;
            curr_node.x += increment_x;
            curr_node.y += increment_y;

            //Double check the pixel is in range of map data
            if (pixel < map_ptr->data.size())
            {
                if (map_ptr->data[pixel] > threshold)
                    return true;
            }

            else
                break;
        }
        return false;
    }

    double VoronoiPath::manhattanDist(const GraphNode &a, const GraphNode &b)
    {
        return fabs(a.x - b.x) + fabs(a.y - b.y);
    }

    double VoronoiPath::euclideanDist(const GraphNode &a, const GraphNode &b)
    {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    }

    int VoronoiPath::getNumberOfNodes()
    {
        return num_nodes;
    }

    bool VoronoiPath::interpolatePaths(std::vector<Path> &paths, double path_waypoint_sep)
    {
        //Calculate minimum distance between two poses in path should have, distance in pix squared
        double waypoint_sep_sq = path_waypoint_sep * path_waypoint_sep / (map_ptr->resolution * map_ptr->resolution);
        for (auto &path : paths)
        {
            for (int i = 1; i < path.path.size(); ++i)
            {
                double sq_dist = pow(path.path[i].x - path.path[i - 1].x, 2) + pow(path.path[i].y - path.path[i - 1].y, 2);

                //Current point and previous point are too far apart, interpolate to get points in between
                if (sq_dist > waypoint_sep_sq)
                {
                    //sqrt steps because sq_dist and waypoint_sep_sq are squared. 1/0.2 != 1/0.04
                    int steps = sqrt(sq_dist / waypoint_sep_sq);
                    GraphNode prev_point = path.path[i - 1];
                    GraphNode curr_point = path.path[i];
                    GraphNode interp_point;
                    int insert_position = i;

                    for (int j = 1; j < steps; ++j)
                    {
                        interp_point.x = prev_point.x * (steps - j) / (double)steps + curr_point.x * (j) / (double)steps;
                        interp_point.y = prev_point.y * (steps - j) / (double)steps + curr_point.y * (j) / (double)steps;

                        //Insert if no collision, .insert will make the nth element your new item
                        if(!edgeCollides(interp_point, interp_point, collision_threshold))
                            path.path.insert(path.path.begin() + (insert_position++), interp_point);

                        //Incrememnt i because a node has been inserted before i
                        ++i;
                    }
                }
            }
        }

        return true;
    }

    bool VoronoiPath::clearPreviousPaths()
    {
        previous_paths.clear();
        previous_path_costs.clear();

        return true;
    }

    bool VoronoiPath::hasPreviousPaths()
    {
        return !previous_paths.empty();
    }

    bool VoronoiPath::isClassDifferent(const std::complex<double> &complex_1, const std::complex<double> &complex_2)
    {
        return std::abs(complex_1 - complex_2) / std::abs(complex_1) > h_class_threshold;
    }

    bool VoronoiPath::removeExcessBranch(std::vector<std::vector<int>> &new_adj_list, double thresh, int curr_node, int prev_node, double cum_dist)
    {
        //Branch is too long, break premptively
        if (cum_dist >= thresh)
            return false;

        //Reached branch node, check with reference with original unmodified adj_list to prevent excessive pruning esp at corridors
        if (adj_list[curr_node].size() >= 3)
        {
            //Delete the previous node from curr_node's adjacency list
            auto it = std::find(new_adj_list[curr_node].begin(), new_adj_list[curr_node].end(), prev_node);
            if (it != new_adj_list[curr_node].end())
                new_adj_list[curr_node].erase(it);

            //Return true indicating branch has been reached
            return true;
        }

        //Traverse all nodes connected to the current one
        for (const auto &connected_node : new_adj_list[curr_node])
        {
            //Skip traversing where we came from
            if (connected_node == prev_node)
                continue;

            //Get distance from curr_node to next node
            double dist = sqrt(pow(node_inf[connected_node].x - node_inf[curr_node].x, 2) +
                                pow(node_inf[connected_node].y - node_inf[curr_node].y, 2));

            //If branch node was found before reaching distance limit, then remove all adjacencies of curr_node, this branch is dead
            if (removeExcessBranch(new_adj_list, thresh, connected_node, curr_node, dist + cum_dist))
            {
                new_adj_list[curr_node].clear();
                return true;
            }
        }

        //Should not reach here
        return false;
    }


    bool VoronoiPath::liesInSquare(const GraphNode & point, const GraphNode & line_point_a, const GraphNode & line_point_b)
    {
        //Checks if the point lies in a square form
        return (point.x - line_point_a.x) * (point.x - line_point_b.x) < 0.0 && (point.y - line_point_a.y) * (point.y - line_point_b.y) < 0.0;
    }
} // namespace voronoi_path
