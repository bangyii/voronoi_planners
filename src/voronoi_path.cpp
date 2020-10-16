#include <voronoi_path.h>
#include <iostream>
#include <algorithm>
#include <exception>
#include <future>
#include <thread>
#include <functional>
#include <cmath>

namespace voronoi_path
{
    voronoi_path::voronoi_path()
    {
    }

    void voronoi_path::setLocalVertices(const std::vector<GraphNode> &vertices)
    {
        local_vertices = vertices;
    }

    std::vector<std::complex<double>> voronoi_path::findObstacleCentroids()
    {
        if (map_ptr->data.size() != 0)
        {
            auto copy_time = std::chrono::system_clock::now();

            cv::Mat cv_map = cv::Mat(map_ptr->data).reshape(0, map_ptr->height);
            cv_map.convertTo(cv_map, CV_8UC1);

            //Downscale to increase contour finding speed
            cv::resize(cv_map, cv_map, cv::Size(), open_cv_scale, open_cv_scale);
            cv::flip(cv_map, cv_map, 1);
            cv::transpose(cv_map, cv_map);
            cv::flip(cv_map, cv_map, 1);
            if (print_timings)
                std::cout << "Time to copy map data " << (std::chrono::system_clock::now() - copy_time).count() / 1000000000.0 << std::endl;

            auto start_time = std::chrono::system_clock::now();

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::Canny(cv_map, cv_map, 50, 150, 3);
            cv::findContours(cv_map, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

            //Get center of centroids
            std::vector<cv::Moments> mu(contours.size());
            centers = std::vector<std::complex<double>>(contours.size());
            for (int i = 0; i < contours.size(); ++i)
            {
                mu[i] = moments(contours[i], false);
                centers[i] = std::complex<double>(map_ptr->width - mu[i].m01 / mu[i].m00 / open_cv_scale, map_ptr->height - mu[i].m10 / mu[i].m00 / open_cv_scale);
            }

            //Delete NaN centroids
            auto it = centers.begin();
            while (it != centers.end())
            {
                if (std::isnan(it->real()) || std::isnan(it->imag()))
                {
                    it = centers.erase(it);
                    continue;
                }

                ++it;
            }

            double a = (centers.size() - 1) / 2.0;
            double b = a;
            obs_coeff.clear();
            obs_coeff.resize(centers.size());

            std::complex<double> from_begin(1,1);
            std::complex<double> from_end(1,1);
            // std::complex<double> al_denom_prod(1, 1);
            for(int i = 0; i < centers.size(); ++i)
            {
                obs_coeff[i] *= from_begin;
                from_begin *= centers[i];

                obs_coeff[centers.size() - i - 1] *= from_end;
                from_end *= centers[centers.size() - i - 1];
            }

            if (print_timings)
                std::cout << "Time to find contour " << (std::chrono::system_clock::now() - start_time).count() / 1000000000.0 << std::endl;
        }

        return centers;
    }

    std::vector<jcv_point> voronoi_path::fillOccupancyVector(const int &start_index, const int &num_pixels)
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

    bool voronoi_path::mapToGraph(Map* map_ptr_)
    // bool voronoi_path::mapToGraph(const Map &map_)
    {
        auto start_time = std::chrono::system_clock::now();

        map_ptr = map_ptr_;

        //Lock mutex to ensure adj_list is not being used        
        auto lock_start = std::chrono::system_clock::now();
        std::lock_guard<std::mutex> lock(voronoi_mtx);

        if(print_timings)
            std::cout << "Map to graph lock duration: " << (std::chrono::system_clock::now() - lock_start).count()/1000000000.0 << "\n";


        int size = map_ptr->data.size();
        if (size == 0)
            return false;

        //Set bottom left and top right for use during homotopy check
        BL = std::complex<double>(0, 0);
        TR = std::complex<double>(map_ptr->width - 1, map_ptr->height - 1);

        //Reset all variables
        edge_vector.clear();
        adj_list.clear();
        node_inf.clear();

        auto loop_map_points = std::chrono::system_clock::now();
        // Loop through map to find occupied cells
        std::vector<jcv_point> points_vec;

        int num_threads = std::thread::hardware_concurrency();

        std::vector<std::future<std::vector<jcv_point>>> future_vector;
        future_vector.reserve(num_threads - 1);

        int num_pixels = floor(size / num_threads);
        int start_pixel = 0;

        for (int i = 0; i < num_threads - 1; ++i)
        {
            start_pixel = i * num_pixels;
            future_vector.emplace_back(std::async(std::launch::async, &voronoi_path::fillOccupancyVector, this, start_pixel, num_pixels));
        }

        //For last thread, take all remaining pixels
        //This current thread is the nth thread
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

        if (print_timings)
            std::cout << "Number of occupied points: " << occupied_points << std::endl;

        jcv_point *points = (jcv_point *)malloc(occupied_points * sizeof(jcv_point));

        if (!points)
        {
            std::cout << "Failed to allocate memory for points array" << std::endl;
            return false;
        }

        for (int i = 0; i < occupied_points; ++i)
            points[i] = points_vec[i];

        if (print_timings)
            std::cout << "Loop map points: \t" << (std::chrono::system_clock::now() - loop_map_points).count() / 1000000000.0 << "s\n";

        //Set the minimum and maximum bounds for voronoi diagram. Follows size of map
        jcv_rect rect;
        rect.min.x = 0;
        rect.min.y = 0;
        rect.max.x = map_ptr->width - 1;
        rect.max.y = map_ptr->height - 1;

        auto diagram_time = std::chrono::system_clock::now();
        jcv_diagram diagram;
        memset(&diagram, 0, sizeof(jcv_diagram));

        //Tried diagram generation in another thread, does not help
        jcv_diagram_generate(occupied_points, points, &rect, 0, &diagram);

        //Get edges from voronoi diagram
        const jcv_edge *edges = jcv_diagram_get_edges(&diagram);
        while (edges)
        {
            edge_vector.push_back(edges);
            edges = jcv_diagram_get_next_edge(edges);
        }

        if (print_timings)
            std::cout << "Generating edges: \t " << (std::chrono::system_clock::now() - diagram_time).count() / 1000000000.0 << "s\n";

        auto clearing_time = std::chrono::system_clock::now();

        //Remove edge vertices that are in obtacle
        removeObstacleVertices();

        //Remove edges that pass through obstacle
        removeCollisionEdges();
        if (print_timings)
            std::cout << "Clearing edges: \t" << (std::chrono::system_clock::now() - clearing_time).count() / 1000000000.0 << "s\n";

        auto adj_list_time = std::chrono::system_clock::now();

        //Convert edges to adjacency list
        std::unordered_map<uint32_t, int> hash_index_map;
        for (int i = 0; i < edge_vector.size(); ++i)
        {
            //Get hash for both vertices of the current edge
            std::vector<uint32_t> hash_vec = {hash(edge_vector[i]->pos[0].x, edge_vector[i]->pos[0].y), hash(edge_vector[i]->pos[1].x, edge_vector[i]->pos[1].y)};
            int node_index[2] = {-1, -1};

            //Check if each node is already in the map
            for (int j = 0; j < hash_vec.size(); ++j)
            {
                auto node_it = hash_index_map.find(hash_vec[j]);

                //Node already exists
                if (node_it != hash_index_map.end())
                    node_index[j] = node_it->second;

                else
                {
                    //Add new node to adjacency list & info vector, and respective hash and node index to map
                    node_index[j] = adj_list.size();
                    node_inf.emplace_back(edge_vector[i]->pos[j].x, edge_vector[i]->pos[j].y);
                    adj_list.push_back(std::vector<int>());
                    hash_index_map.insert(std::pair<uint32_t, int>(hash_vec[j], node_index[j]));
                }
            }

            //Once both node indices are found, add edge between the two nodes
            adj_list[node_index[0]].push_back(node_index[1]);
            adj_list[node_index[1]].push_back(node_index[0]);
        }

        //Connect single edges to nearby node if <= node_connection_threshold_pix pixel distance
        int threshold = pow(node_connection_threshold_pix, 2);
        for(int i = 0; i < adj_list.size(); ++i)
        {
            //Singly connected node
            if(adj_list[i].size() == 1)
            {
                //Check through all node_inf to see if there are any within distance threshold
                for(int j = 0 ; j < node_inf.size(); ++j)
                {
                    //If the node being checked is i or is already connected to i
                    if(j == i || std::find(adj_list[i].begin(), adj_list[i].end(), j) != adj_list[i].end())
                        continue; 
                        
                    double dist = pow(node_inf[j].x - node_inf[i].x, 2) + pow(node_inf[j].y - node_inf[i].y, 2);
                    
                    if(dist <= threshold)
                    {
                        adj_list[i].push_back(j);
                        adj_list[j].push_back(i);
                        break;
                    }
                }
            }
        }

        if (print_timings)
        {
            std::cout << "Number of nodes: " << adj_list.size() << std::endl;
            std::cout << "Adjacency list: \t " << (std::chrono::system_clock::now() - adj_list_time).count() / 1000000000.0 << "s\n";
            std::cout << "Convert to edges: \t" << ((std::chrono::system_clock::now() - start_time).count() / 1000000000.0) << "s\n";
        }

        //Get centroids after map has been updated
        findObstacleCentroids();

        num_nodes = adj_list.size();

        jcv_diagram_free(&diagram);
        free(points);
        return true;
    }

    std::vector<std::vector<int>> voronoi_path::getAdjList()
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        return adj_list;
    }

    bool voronoi_path::getObstacleCentroids(std::vector<GraphNode> &centroids)
    {
        centroids.reserve(centers.size());
        for(const auto &elem : centers)
            centroids.emplace_back(elem.real(), elem.imag());

        return true;
    }

    bool voronoi_path::getEdges(std::vector<GraphNode> &edges)
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
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

    bool voronoi_path::getDisconnectedNodes(std::vector<GraphNode> &nodes)
    {
        std::lock_guard<std::mutex> lock(voronoi_mtx);
        for(int i = 0; i < num_nodes; ++i)
        {
            //If the node is only connected on one side
            if(adj_list[i].size() == 1)
                nodes.emplace_back(node_inf[i].x, node_inf[i].y);
        }

        return true;
    }

    void voronoi_path::printEdges()
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

    uint32_t voronoi_path::hash(const double &x, const double &y)
    {
        uint32_t hashed_int = std::hash<uint32_t>{}(static_cast<uint32_t>((static_cast<uint16_t>(x) << 16) ^ static_cast<uint16_t>(y)));
        return hashed_int;
    }

    bool voronoi_path::trimPathBeginning(std::vector<GraphNode> &path)
    {           
        std::vector<GraphNode> path_copy = path;
        double cum_distance_sq = 0;
        int i = 1;
        for(i = 1; i < path.size() - 1; ++i)
        {
            cum_distance_sq += pow(path[i].x - path[i-1].x, 2) + pow(path[i].y - path[i-1].y, 2);

            if(edgeCollides(path[0], path[i+1]))
                break;
            
        }

        auto delete_it = path.begin() + 1;
        while(i > 1)
        {                      
            delete_it = path.erase(delete_it);
            --i;
        }

        //Find the halfpoint distance of the original path and then add that node to ensure smooth path
        double half_distance = 0;
        for(i = 1; i < path_copy.size(); ++i)
        {
            half_distance += pow(path_copy[i].x - path_copy[i-1].x, 2) + pow(path_copy[i].y - path_copy[i-1].y, 2);

            if(half_distance > cum_distance_sq/4.0)
                break;
        }

        return true;
    }

    std::vector<std::vector<GraphNode>> voronoi_path::getPath(const GraphNode &start, const GraphNode &end, const int &num_paths)
    {
        //Block until voronoi is no longer being updated. Prevents issue where planning is done using an empty adjacency list
        auto lock_start = std::chrono::system_clock::now();
        std::lock_guard<std::mutex> lock(voronoi_mtx);

        if(print_timings)
            std::cout << "Get path lock duration: " << (std::chrono::system_clock::now() - lock_start).count()/1000000000.0 << "\n";

        auto start_time = std::chrono::system_clock::now();
        std::vector<std::vector<GraphNode>> path;

        int start_node, end_node;
        if (!getNearestNode(start, end, start_node, end_node))
            return path;

        std::vector<int> shortest_path;
        double cost;
        auto shortest_time = std::chrono::system_clock::now();
        if (findShortestPath(start_node, end_node, shortest_path, cost))
        {
            if (print_timings)
                std::cout << "Find shortest path: \t" << ((std::chrono::system_clock::now() - shortest_time).count() / 1000000000.0) << "s\n";

            std::vector<std::vector<int>> all_paths;
            auto kth_time = std::chrono::system_clock::now();
            //Get next shortest path
            if (num_paths >= 1)
            {
                if (print_timings)
                    std::cout << "Finding alternate paths\n";

                kthShortestPaths(start_node, end_node, shortest_path, all_paths, num_paths - 1);
            }

            if (print_timings)
                std::cout << "Find alternate paths: \t" << ((std::chrono::system_clock::now() - kth_time).count() / 1000000000.0) << "s\n";

            auto p_process_start = std::chrono::system_clock::now();
            //Copy all_paths into new container which include start and end
            std::vector<std::vector<GraphNode>> all_path_nodes;
            all_path_nodes.reserve(all_paths.size());

            for (int i = 0; i < all_paths.size(); ++i)
            {
                all_path_nodes.push_back(std::vector<GraphNode>{start});
                all_path_nodes[i].reserve(all_paths[i].size() + 2);

                for(const auto& node : all_paths[i])
                    all_path_nodes[i].emplace_back(node_inf[node].x, node_inf[node].y);

                all_path_nodes[i].push_back(end);

                //Trim beginning of path to remove unnecessary u-turns in path
                if(trim_path_beginning)
                    trimPathBeginning(all_path_nodes[i]);
            }

            //Bezier interpolation
            //TODO: Can be threaded as well
            //TODO: Shorten paths generated by bypassing nodes that do not help to avoid collision
            //For all paths, j = path number
            for (int j = 0; j < all_path_nodes.size(); ++j)
            {
                std::vector<GraphNode> bezier_path;
                int num_of_nodes = all_path_nodes[j].size();

                std::vector<GraphNode> sub_nodes;
                std::vector<GraphNode> prev_2_nodes;

                //For all nodes in path
                for (int i = 1; i < num_of_nodes; ++i)
                {
                    //If adjacent edges in original path collide, then something is wrong with map
                    //Return empty path because the original path has collision, not feasible
                    if(edgeCollides(all_path_nodes[j][i-1], all_path_nodes[j][i]))
                        return std::vector<std::vector<GraphNode>>();

                    //Add previous node and extra node if sub_nodes was recently reset due to collision or initialization
                    if (sub_nodes.size() == 0)
                    {
                        sub_nodes.push_back(all_path_nodes[j][i - 1]);

                        //Calculate extra node based on previous subsection's gradient
                        if (i > 1 && prev_2_nodes.size() == 2)
                        {
                            GraphNode dir(prev_2_nodes[1] - prev_2_nodes[0]);
                            dir.setUnitVector();

                            //Extra point is collinear with prev[0] and prev[1], but further along than p[1]
                            sub_nodes.push_back(prev_2_nodes[1] + dir * extra_point_distance * map_ptr->resolution);

                            //Do not insert if collision occurs when extra point is added
                            if (edgeCollides(sub_nodes[sub_nodes.size() - 2], sub_nodes.back()))
                                sub_nodes.pop_back();

                            prev_2_nodes.clear();
                        }
                    }

                    //If this node to the first node does not collide
                    if (!edgeCollides(sub_nodes[0], all_path_nodes[j][i]) && sub_nodes.size() < bezier_max_n)
                        sub_nodes.push_back(all_path_nodes[j][i]);

                    //Collision happened or limit reached, find sub path with current sub nodes
                    else
                    {
                        //Retrace back i value to prevent skipping a node
                        --i;

                        //Calculate the bezier subsection
                        std::vector<GraphNode> temp_bezier = bezierSubsection(sub_nodes);
                        bezier_path.insert(bezier_path.end(), temp_bezier.begin(), temp_bezier.end());

                        if (sub_nodes.size() > 1)
                            prev_2_nodes.insert(prev_2_nodes.begin(), sub_nodes.end() - 2, sub_nodes.end());

                        sub_nodes.clear();
                    }
                }

                //If no collision before the end, find sub path as well
                if (sub_nodes.size() != 0)
                {
                    std::vector<GraphNode> temp_bezier = bezierSubsection(sub_nodes);
                    bezier_path.insert(bezier_path.end(), temp_bezier.begin(), temp_bezier.end());
                    sub_nodes.clear();
                }
                path.push_back(bezier_path);
                // std::cout << "Bezier path size " << bezier_path.size() << "\n";
            }

            if (print_timings)
            {
                std::cout << "Post process all paths: " << (std::chrono::system_clock::now() - p_process_start).count() / 1000000000.0 << "s\n";
                std::cout << "Find all paths, including time to find nearest node: \t" << ((std::chrono::system_clock::now() - start_time).count() / 1000000000.0) << "s\n";
            }
        }

        else
            std::cout << "Path could not be found" << std::endl;

        return path;
    }

    bool voronoi_path::getNearestNode(const GraphNode &start, const GraphNode &end, int &start_node, int &end_node)
    {
        auto start_time = std::chrono::system_clock::now();
        //TODO: Should not only check nearest nodes. Should allow nearest position to be on an edge

        double min_start_dist = std::numeric_limits<double>::infinity();
        double min_end_dist = std::numeric_limits<double>::infinity();
        start_node = -1;
        end_node = -1;

        double ang, temp_end_dist, temp_start_dist;
        GraphNode curr;

        for (int i = 0; i < num_nodes; ++i)
        {
            curr.x = node_inf[i].x;
            curr.y = node_inf[i].y;


            //If potential starting node brings robot towards end goal
            temp_start_dist = pow(curr.x - start.x, 2) + pow(curr.y - start.y, 2);
            if (temp_start_dist < min_start_dist)
            {
                if (!edgeCollides(start, curr))
                {
                    min_start_dist = temp_start_dist;
                    start_node = i;
                }
            }

            temp_end_dist = pow(curr.x - end.x, 2) + pow(curr.y - end.y, 2);
            if (temp_end_dist < min_end_dist)
            {
                if (!edgeCollides(end, curr))
                {
                    min_end_dist = temp_end_dist;
                    end_node = i;
                }
            }

        }

        //Failed to find start/end even after relaxation
        if (start_node == -1 || end_node == -1)
        {
            std::cout << "Failed to find nearest starting or ending node" << std::endl;
            return false;
        }

        if (print_timings)
            std::cout << "Find nearest node: \t" << ((std::chrono::system_clock::now() - start_time).count() / 1000000000.0) << "s\n";

        return true;
    }

    //https://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-216.pdf
    std::complex<double> voronoi_path::calcHomotopyClass(const std::vector<int> &path_)
    {
        std::vector<std::complex<double>> path;
        path.reserve(path_.size());

        //Convert path to complex path
        for (auto node : path_)
            path.emplace_back(node_inf[node].x, node_inf[node].y);

        //Go through each edge of the path
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
                [&, start_pose, poses_per_thread, path](const std::vector<std::complex<double>> &centers) {
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
                },
                std::ref(centers)));
        }

        for (int i = 0; i < future_vector.size(); ++i)
        {
            future_vector[i].wait();
            std::complex<double> temp_sum = future_vector[i].get();
            path_sum += temp_sum;
        }
        return path_sum;
    }

    bool voronoi_path::kthShortestPaths(const int &start_node, const int &end_node, const std::vector<int> &shortestPath, std::vector<std::vector<int>> &all_paths, const int &num_paths)
    {
        double adjacency_cum_time = 0;
        double copy_root_cum_time = 0;
        double disconnect_edge_cum_time = 0;
        double remove_nodes_cum_time = 0;
        double spur_path_cum_time = 0;
        double copy_kth_cum_time = 0;
        double get_total_cost_time = 0;
        double calc_homotopy_cum_time = 0;
        double check_uniqueness_cum_time = 0;
        double check_min_homotopy_cum_time = 0;

        all_paths.reserve(num_paths + 1);

        if (num_paths == 0)
        {
            all_paths.push_back(shortestPath);
            return true;
        }

        std::vector<std::vector<int>> kthPaths;
        kthPaths.reserve(num_paths + 1);
        kthPaths.push_back(shortestPath);

        std::vector<std::vector<int>> adj_list_backup(adj_list);
        std::vector<int> adj_list_modified_ind;

        std::vector<std::vector<int>> potentialKth;
        std::vector<std::pair<double, int>> cost_index_vec;
        std::vector<std::complex<double>> homotopy_classes;

        for (int k = 1; k <= num_paths; ++k)
        {
            //Break if cannot find the previous k path, could not find all NUM_PATHS
            if (k - 1 == kthPaths.size())
                break;

            auto calc_homo_start = std::chrono::system_clock::now();
            //Update homotopy classes vector whenever new kth path gets added
            homotopy_classes.push_back(std::move(calcHomotopyClass(kthPaths.back())));
            calc_homotopy_cum_time += (std::chrono::system_clock::now() - calc_homo_start).count() / 1000000000.0;

            //Spur node is ith node, from start to 2nd last node of path, inclusive
            for (int i = 0; i < kthPaths[k - 1].size() - 1; ++i)
            {
                int spurNode = kthPaths[k - 1][i];

                //Copy root path into container. Root path is path up until spur node, containing the path from start onwards
                auto copy_root_time = std::chrono::system_clock::now();
                std::vector<int> rootPath(i + 1);
                std::copy(kthPaths[k - 1].begin(), kthPaths[k - 1].begin() + i + 1, rootPath.begin());
                copy_root_cum_time += (std::chrono::system_clock::now() - copy_root_time).count() / 1000000000.0;

                //Disconnect edges if root path has already been discovered before
                auto disconnect_edges_time = std::chrono::system_clock::now();
                for (auto paths : kthPaths)
                {
                    int equal_count = 0;
                    for (int z = 0; z <= i; ++z)
                    {
                        if (z >= paths.size() || z >= rootPath.size())
                            break;

                        if (paths[z] == rootPath[z])
                            equal_count++;
                    }

                    //Entire root path is identical
                    if (equal_count == rootPath.size() && i + 1 < paths.size())
                    {
                        //Remove edge from spurNode to spur_next
                        int spur_next = paths[i + 1];
                        auto erase_it = std::find(adj_list[spurNode].begin(), adj_list[spurNode].end(), spur_next);

                        //Edge not removed yet
                        if (erase_it != adj_list[spurNode].end())
                        {
                            adj_list[spurNode][erase_it - adj_list[spurNode].begin()] = -1;
                            adj_list_modified_ind.push_back(spurNode);
                        }

                        //Remove edge from spur_next to spurNode
                        erase_it = std::find(adj_list[spur_next].begin(), adj_list[spur_next].end(), spurNode);

                        //Edge not removed yet
                        if (erase_it != adj_list[spur_next].end())
                        {
                            adj_list[spur_next][erase_it - adj_list[spur_next].begin()] = -1;
                            adj_list_modified_ind.push_back(spur_next);
                        }
                    }
                }
                disconnect_edge_cum_time += (std::chrono::system_clock::now() - disconnect_edges_time).count() / 1000000000.0;

                //Remove all nodes of root path from graph except spur node and start node
                auto node_start_time = std::chrono::system_clock::now();

                //Exclude spurNode (rootPath.back())
                for (int node_ind = 0; node_ind < rootPath.size() - 1; ++node_ind)
                {
                    int node = rootPath[node_ind];

                    for (int del_ind = 0; del_ind < adj_list[node].size(); ++del_ind)
                    {
                        //Remove connection from point-er side
                        int pointed_to = adj_list[node][del_ind];

                        //Edge is not deleted
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
                remove_nodes_cum_time += (std::chrono::system_clock::now() - node_start_time).count() / 1000000000.0;

                std::vector<int> spur_path;
                double cost;

                //if spur path is found
                auto find_spur_path_start = std::chrono::system_clock::now();
                if (findShortestPath(spurNode, end_node, spur_path, cost))
                {
                    spur_path_cum_time += (std::chrono::system_clock::now() - find_spur_path_start).count() / 1000000000.0;

                    std::vector<int> total_path;

                    auto check_unique_start = std::chrono::system_clock::now();
                    if (rootPath.size())
                        total_path.insert(total_path.begin(), rootPath.begin(), rootPath.end() - 1);
                    total_path.insert(total_path.end(), spur_path.begin(), spur_path.end());

                    //Check if the path just generated is unique in the potential k paths
                    bool path_is_unique = true;

                    //Make sure to check all paths
                    int last = potentialKth.size();
                    if (kthPaths.size() > potentialKth.size())
                        last = kthPaths.size();

                    int kth_equal, pot_equal;
                    for (int check_path = 0; check_path < last; ++check_path)
                    {
                        kth_equal = 0;
                        pot_equal = 0;

                        for (int node_pot = 0; node_pot < total_path.size(); ++node_pot)
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
                                if (node_pot < potentialKth[check_path].size())
                                    if (potentialKth[check_path][node_pot] == total_path[node_pot])
                                        pot_equal++;
                            }
                        }

                        if (pot_equal == total_path.size() || kth_equal == total_path.size())
                            path_is_unique = false;
                    }
                    check_uniqueness_cum_time += (std::chrono::system_clock::now() - check_unique_start).count() / 1000000000.0;

                    //add unique path to potentials
                    auto get_total_start = std::chrono::system_clock::now();
                    if (path_is_unique)
                    {
                        //Get cost of total path
                        double total_cost = 0;
                        for (int int_node = 0; int_node < total_path.size() - 1; ++int_node)
                            total_cost += euclideanDist(node_inf[total_path[int_node]], node_inf[total_path[int_node + 1]]);

                        cost_index_vec.push_back(std::make_pair(total_cost, potentialKth.size()));
                        potentialKth.push_back(std::move(total_path));
                    }
                    get_total_cost_time += (std::chrono::system_clock::now() - get_total_start).count() / 1000000000.0;
                }

                auto restore_start = std::chrono::system_clock::now();
                //Reset adj_list before changing spur node
                for(const auto& mod_node : adj_list_modified_ind)
                    adj_list[mod_node] = adj_list_backup[mod_node];

                adj_list_modified_ind.clear();
                adj_list_modified_ind.shrink_to_fit();
                adjacency_cum_time += (std::chrono::system_clock::now() - restore_start).count() / 1000000000.0;
            }

            //No alternate paths found
            if (potentialKth.size() == 0)
                break;

            std::sort(cost_index_vec.begin(), cost_index_vec.end());

            auto it = cost_index_vec.begin();
            while (it != cost_index_vec.end())
            {
                calc_homo_start = std::chrono::system_clock::now();
                std::complex<double> curr_h_class = calcHomotopyClass(potentialKth[it->second]);
                calc_homotopy_cum_time += (std::chrono::system_clock::now() - calc_homo_start).count() / 1000000000.0;

                auto check_min_homo_start = std::chrono::system_clock::now();
                int h = 0;
                for (h = 0; h < homotopy_classes.size(); ++h)
                {
                    //Path is not unique
                    if (std::abs(curr_h_class - homotopy_classes[h]) / std::abs(curr_h_class) <= h_class_threshold)
                    {
                        //Erase and then break if not unique
                        it = cost_index_vec.erase(it);
                        break;
                    }
                }

                //Path is unique
                if (h == homotopy_classes.size())
                    break;
                check_min_homotopy_cum_time += (std::chrono::system_clock::now() - check_min_homo_start).count()/1000000000.0;
            }

            auto copy_kth = std::chrono::system_clock::now();
            if (!cost_index_vec.empty())
            {
                int copy_index = cost_index_vec[0].second;

                kthPaths.push_back(potentialKth[copy_index]);
                cost_index_vec.erase(cost_index_vec.begin());
            }

            copy_kth_cum_time += (std::chrono::system_clock::now() - copy_kth).count() / 1000000000.0;
        }

        all_paths.insert(all_paths.begin(), kthPaths.begin(), kthPaths.end());

        if (print_timings)
        {
            std::cout << "Cum adjacency list restore: " << adjacency_cum_time << "\n";
            std::cout << "Cum copy root path: " << copy_root_cum_time << "\n";
            std::cout << "Cum disconnect edges: " << disconnect_edge_cum_time << "\n";
            std::cout << "Cum remove nodes of root path: " << remove_nodes_cum_time << "\n";
            std::cout << "Cum find spur path: " << spur_path_cum_time << "\n";
            std::cout << "Cum copy kth path: " << copy_kth_cum_time << "\n";
            std::cout << "Cum get total cost: " << get_total_cost_time << "\n";
            std::cout << "Cum calc homotopy: " << calc_homotopy_cum_time << "\n";
            std::cout << "Cum check unique: " << check_uniqueness_cum_time << "\n";
            std::cout << "Cum check min homotopy: " <<  check_min_homotopy_cum_time << "\n";
        }

        if (num_paths == all_paths.size() - 1)
            return true;

        else
            return false;
    }

    bool voronoi_path::findShortestPath(const int &start_node, const int &end_node, std::vector<int> &path, double &cost)
    {
        auto start_time = std::chrono::system_clock::now();
        std::vector<std::pair<int, NodeInfo>> open_list;
        std::vector<bool> nodes_closed_bool(num_nodes, false);
        std::vector<int> nodes_prev(num_nodes, -1);

        NodeInfo start_info;
        start_info.cost_upto_here = 0;
        start_info.cost_to_goal = euclideanDist(node_inf[start_node], node_inf[end_node]);
        start_info.updateCost();
        open_list.emplace_back(std::make_pair(start_node, start_info));

        GraphNode end_node_location = node_inf[end_node];
        GraphNode curr_node_location;
        GraphNode next_node_location;
        NodeInfo curr_node_info;
        int curr_node = start_node;
        int min_ind = 0;
        int next_node;

        //Run until the end_node enters the closed list
        while (!nodes_closed_bool[end_node])
        {
            curr_node = open_list[0].first;
            curr_node_location = node_inf[curr_node];

            //Get info for current node
            auto open_start = std::chrono::system_clock::now();
            curr_node_info = open_list[0].second;
            open_list_time += (std::chrono::system_clock::now() - open_start).count() / 1000000000.0;
            cost = curr_node_info.total_cost;

            //Loop all adjacent nodes of current node
            for (int i = 0; i < adj_list[curr_node].size(); ++i)
            {
                next_node = adj_list[curr_node][i];

                //Edge has been deleted
                if (next_node == -1)
                    continue;

                //If next node is in closed list, skip
                auto closed_start = std::chrono::system_clock::now();

                if (nodes_closed_bool[next_node])
                    continue;

                closed_list_time += (std::chrono::system_clock::now() - closed_start).count() / 1000000000.0;

                //Get the location of the next node
                next_node_location = node_inf[next_node];

                //Calculate cost upto the next node from start node
                double start_to_next_dist = euclideanDist(curr_node_location, next_node_location) + curr_node_info.cost_upto_here;

                //Find next_node in open_list
                auto open_start = std::chrono::system_clock::now();

                auto it = std::find_if(open_list.begin(), open_list.end(),
                                       [&next_node](const std::pair<int, NodeInfo> &in) {
                                           return in.first == next_node;
                                       });

                open_list_time += (std::chrono::system_clock::now() - open_start).count() / 1000000000.0;
                //If node is not in open list yet
                if (it == open_list.end())
                {
                    NodeInfo new_node;
                    new_node.cost_upto_here = start_to_next_dist;
                    new_node.cost_to_goal = euclideanDist(end_node_location, next_node_location);
                    new_node.updateCost();
                    nodes_prev[next_node] = curr_node;

                    open_list.emplace_back(std::make_pair(next_node, new_node));
                }

                else
                {
                    //Update node's nearest distance to reach here, if the new cost is lower
                    if (start_to_next_dist < it->second.cost_upto_here)
                    {
                        //Update cost upto here. Cost to goal doesn't change
                        it->second.cost_upto_here = start_to_next_dist;
                        it->second.updateCost();
                        nodes_prev[next_node] = curr_node;
                    }
                }
            }

            //Remove curr_node from open list after all adjacent nodes have been put into open list
            open_list.erase(open_list.begin());

            //Then put into closed list
            nodes_closed_bool[curr_node] = true;

            // Find minimum total_cost in open_list
            if (!open_list.empty())
            {
                //Sort by pair's second element.total_cost
                std::sort(open_list.begin(), open_list.end(), [](std::pair<int, NodeInfo> &left, std::pair<int, NodeInfo> &right){
                    return left.second.total_cost < right.second.total_cost;
                });
            }

            else
                break;
        }
        find_path_time += (std::chrono::system_clock::now() - start_time).count() / 1000000000.0;

        auto copy_path_start = std::chrono::system_clock::now();
        //Put nodes into path
        std::vector<int> temp_path;
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

        copy_path_time += (std::chrono::system_clock::now() - copy_path_start).count() / 1000000000.0;
        return true;
    }

    void voronoi_path::removeObstacleVertices()
    {
        //Get edge vertices that are in obtacle
        //Data loaded by map server is upside down. Top of image is last of data array
        //Left right order is the same as in image
        //Meaning map.data reads from image from bottom of image, upwards, left to right

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
            std::vector<const jcv_edge*> remaining_edges;
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
    }

    void voronoi_path::removeCollisionEdges()
    {
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

        std::vector<int> delete_indices;
        for (int i = 0; i < edge_vector.size(); ++i)
        {
            const jcv_edge* curr_edge = edge_vector[i];

            GraphNode start(curr_edge->pos[0].x, curr_edge->pos[0].y);
            GraphNode end(curr_edge->pos[1].x, curr_edge->pos[1].y);

            if (edgeCollides(start, end))
                delete_indices.push_back(i);
        }

        if (delete_indices.size() != 0)
        {
            std::vector<const jcv_edge*> remaining_edges;
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
    }

    double voronoi_path::vectorAngle(const double vec1[2], const double vec2[2])
    {
        double dot = vec1[0] * vec2[0] + vec1[1] * vec2[1];
        double det = vec1[0] * vec2[1] - vec1[1] * vec2[0];
        return std::atan2(det, dot);
    }

    bool voronoi_path::edgeCollides(const GraphNode &start, const GraphNode &end)
    {
        double steps = 0;
        double distance = sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2));
        double curr_x, curr_y;
        int pixel;

        if (distance > line_check_resolution)
            steps = distance / line_check_resolution;

        else
            steps = 1;

        double increment = 1.0 / steps;
        double curr_step = 0.0;

        //TODO: To double check to ensure curr_step always reaches 1.0
        while (curr_step <= 1.0)
        {
            curr_x = (1.0 - curr_step) * start.x + curr_step * end.x;
            curr_y = (1.0 - curr_step) * start.y + curr_step * end.y;

            pixel = int(curr_x) + int(curr_y) * map_ptr->width;
            if (map_ptr->data[pixel] > collision_threshold)
            {
                return true;
            }

            curr_step += increment;
        }
        return false;
    }

    double voronoi_path::manhattanDist(const GraphNode &a, const GraphNode &b)
    {
        return fabs(a.x - b.x) + fabs(a.y - b.y);
    }

    double voronoi_path::euclideanDist(const GraphNode &a, const GraphNode &b)
    {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    }

    int voronoi_path::getNumberOfNodes()
    {
        return num_nodes;
    }

    int voronoi_path::binomialCoeff(const int &n, const int &k_)
    {
        //https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
        int res = 1;
        int k = k_;

        // Since C(n, k) = C(n, n-k)
        if (k > n - k)
            k = n - k;

        // Calculate value of
        // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
        for (int i = 0; i < k; ++i)
        {
            res *= (n - i);
            res /= (i + 1);
        }

        return res;
    }

    std::vector<GraphNode> voronoi_path::bezierSubsection(std::vector<GraphNode> &points)
    {
        if (points.size() == 1)
            return std::vector<GraphNode>{points[0]};

        //Set first point as reference point
        double curr_x = points.begin()->x;
        double curr_y = points.begin()->y;
        double pixel_threshold = min_node_sep_sq * map_ptr->resolution;

        // // Connect start node to node right before the node that will collide if connected to
        // auto start_trim_it = points.begin() + 1;
        // while(start_trim_it != points.end())
        // {
        //     //If doesn't collide, delete node
        //     if(!edgeCollides(*points.begin(), *start_trim_it))
        //         start_trim_it = points.erase(start_trim_it);

        //     else
        //         break; 
        // }

        // curr_x = points.begin()->x;
        // curr_y = points.begin()->y;

        //Delete points that are too near to each other, starting from 2nd node
        auto it = points.begin() + 1;
        while (it < points.end())
        {
            //If point is too close, delete then move to next node
            double dist = pow(it->x - curr_x, 2) + pow(it->y - curr_y, 2);
            if (dist < pixel_threshold && (it != points.end() - 1))
            {
                it = points.erase(it);
                continue;
            }

            //Else point was not too far, set as new reference point then increment iterator
            else
            {
                curr_x = it->x;
                curr_y = it->y;
                ++it;
            }        
        }

        int n = points.size() - 1;
        std::vector<int> combos(n + 1);
        std::vector<GraphNode> bezier_path;

        //Calculate all required nCk
        for (int i = 0; i < n + 1; ++i)
            combos[i] = binomialCoeff(n, i);

        //20 points bezier interpolation
        for (double t = 0; t <= 1.0 + 0.01; t += 0.05)
        {
            if (t > 1.0)
                t = 1.0;

            GraphNode sum_point;

            //P_i
            for (int i = 0; i <= n; ++i)
            {
                sum_point += points[i] * combos[i] * pow(1 - t, n - i) * pow(t, i);
            }

            bezier_path.push_back(sum_point);
        }

        return bezier_path;
    }

} // namespace voronoi_path
