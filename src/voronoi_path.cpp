#include <voronoi_path.h>
#include <iostream>
#include <algorithm>
#include <exception>
#include <future>
#include <thread>

namespace voronoi_path
{
    voronoi_path::voronoi_path() : updating_voronoi(false), is_planning(false)
    {
    }

    bool voronoi_path::isUpdatingVoronoi()
    {
        return updating_voronoi;
    }

    void voronoi_path::setLocalVertices(const std::vector<GraphNode> &vertices)
    {
        local_vertices = vertices;
    }

    std::vector<std::complex<double>> voronoi_path::findObstacleCentroids()
    {
        // cv::Mat cv_map(map.height, map.width, CV_32SC1);

        if (map.data.size() != 0)
        {
            auto copy_time = std::chrono::system_clock::now();

            cv::Mat cv_map = cv::Mat(map.data).reshape(0, map.height);
            cv_map.convertTo(cv_map, CV_8UC1);

            //Downscale to increase contour finding speed
            cv::resize(cv_map, cv_map, cv::Size(), open_cv_scale, open_cv_scale);
            cv::flip(cv_map, cv_map, 1);
            cv::transpose(cv_map, cv_map);
            cv::flip(cv_map, cv_map, 1);
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

                //Centroids in terms of pixels on the original image
                centers[i] = std::complex<double>(-mu[i].m01 / mu[i].m00 / open_cv_scale, -mu[i].m10 / mu[i].m00 / open_cv_scale);
            }

            //Delete NaN centroids
            auto it = centers.begin();
            while (it != centers.end())
            {
                if (isnan(it->real()) || isnan(it->imag()))
                {
                    it = centers.erase(it);
                    continue;
                }

                ++it;
            }

            // std::cout << "Center 1 : " << centers[0].x << "\t" << centers[0].y << "\n";

            // cv::Mat drawing(cv_map.size(), CV_8UC3, cv::Scalar(255, 255, 255));
            // for (int i = 0; i < contours.size(); ++i)
            // {
            //     drawContours(drawing, contours, i, cv::Scalar(0, 0, 0), 2, 8, hierarchy, 0, cv::Point());
            //     circle(drawing, mc[i], 4, cv::Scalar(0, 0, 255), -1, 8, 0);
            // }

            // cv::imshow("window", drawing);
            // cv::waitKey(0);
            //TODO: Upsize back contour centers

            std::cout << "Time to find contour " << (std::chrono::system_clock::now() - start_time).count() / 1000000000.0 << std::endl;
        }

        return centers;
    }

    //TODO: Pass by ref?
    std::vector<jcv_point> voronoi_path::fillOccupancyVector(const int &start_index, const int &num_pixels)
    {
        //TODO: Reserve num_pixels for points_vec? Could end up using more memory
        std::vector<jcv_point> points_vec;
        for (int i = start_index; i < start_index + num_pixels; i += (pixels_to_skip + 1))
        {
            //Occupied
            if (map.data[i] >= occupancy_threshold)
            {
                jcv_point temp_point;
                temp_point.x = i % map.width;
                temp_point.y = (int)(i / map.width);

                //TODO: use move? Old point no longer required. Emplace back makes no difference
                points_vec.push_back(temp_point);
            }
        }

        return points_vec;
    }

    bool voronoi_path::mapToGraph(const Map &map_)
    {
        auto start_time = std::chrono::system_clock::now();
        updating_voronoi = true;
        //Reset all variables
        map = map_;
        edge_vector.clear();
        adj_list.clear();
        node_inf.clear();

        //Set bottom left and top right for use during homotopy check
        BL = std::complex<double>(0, 0);
        TR = std::complex<double>(map.width - 1, map.height - 1);

        int size = map.data.size();
        if (size == 0 || is_planning)
        {
            updating_voronoi = false;
            return false;
        }

        auto loop_map_points = std::chrono::system_clock::now();
        // Loop through map to find occupied cells
        //TODO: Reserve?
        std::vector<jcv_point> points_vec;

        int num_threads = std::thread::hardware_concurrency();

        //TODO: Reserve
        std::vector<std::future<std::vector<jcv_point>>> future_vector;

        //TODO: Create vector for storing jcv_point vectors from thread
        int num_pixels = floor(size / num_threads);
        int start_pixel = 0;

        for (int i = 0; i < num_threads - 1; ++i)
        {
            start_pixel = i * num_pixels;
            future_vector.emplace_back(std::async(std::launch::async, &voronoi_path::fillOccupancyVector, this, start_pixel, num_pixels));
        }

        //For last thread, take all remaining pixels
        //This current thread is the nth thread
        //TODO: Pass ref?
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
                updating_voronoi = false;
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
        std::cout << "Number of occupied points: " << occupied_points << std::endl;
        jcv_point *points = (jcv_point *)malloc(occupied_points * sizeof(jcv_point));

        if (!points)
        {
            std::cout << "Failed to allocate memory for points array" << std::endl;
            updating_voronoi = false;
            return false;
        }

        for (int i = 0; i < occupied_points; ++i)
        {
            points[i] = points_vec[i];
        }
        if (print_timings)
            std::cout << "Loop map points took " << (std::chrono::system_clock::now() - loop_map_points).count() / 1000000000.0 << "s\n";

        //Set the minimum and maximum bounds for voronoi diagram. Follows size of map
        jcv_rect rect;
        rect.min.x = 0;
        rect.min.y = 0;
        rect.max.x = map.width - 1;
        rect.max.y = map.height - 1;

        auto diagram_time = std::chrono::system_clock::now();
        jcv_diagram diagram;
        memset(&diagram, 0, sizeof(jcv_diagram));

        //Tried diagram generation in another thread, does not help
        jcv_diagram_generate(occupied_points, points, &rect, 0, &diagram);

        //Get edges from voronoi diagram
        const jcv_edge *edges = jcv_diagram_get_edges(&diagram);
        while (edges)
        {
            edge_vector.push_back(*edges);
            edges = jcv_diagram_get_next_edge(edges);
        }

        jcv_diagram_free(&diagram);
        free(points);
        if (print_timings)
            std::cout << "Generating edges took " << (std::chrono::system_clock::now() - diagram_time).count() / 1000000000.0 << "s\n";

        auto clearing_time = std::chrono::system_clock::now();

        //Remove edge vertices that are in obtacle
        removeObstacleVertices();

        //Remove edges that pass through obstacle
        removeCollisionEdges();
        if (print_timings)
            std::cout << "Clearing edges took " << (std::chrono::system_clock::now() - clearing_time).count() / 1000000000.0 << "s\n";

        auto adj_list_time = std::chrono::system_clock::now();
        //Convert edges to adjacency list
        std::map<std::string, int> hash_index_map;
        for (int i = 0; i < edge_vector.size(); ++i)
        {
            //Get hash for both vertices of the current edge
            std::vector<std::string> hash_vec = {hash(edge_vector[i].pos[0].x, edge_vector[i].pos[0].y), hash(edge_vector[i].pos[1].x, edge_vector[i].pos[1].y)};
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
                    node_inf.emplace_back(edge_vector[i].pos[j].x, edge_vector[i].pos[j].y);
                    adj_list.push_back(std::vector<int>());
                    hash_index_map.insert(std::pair<std::string, int>(hash_vec[j], node_index[j]));
                }
            }

            //Once both node indices are found, add edge between the two nodes
            adj_list[node_index[0]].push_back(node_index[1]);
            adj_list[node_index[1]].push_back(node_index[0]);
        }
        std::cout << "Number of nodes: " << adj_list.size() << std::endl;

        if (print_timings)
        {
            std::cout << "Adjacency list took " << (std::chrono::system_clock::now() - adj_list_time).count() / 1000000000.0 << "s\n";
            std::cout << "Time taken to convert to edges: " << ((std::chrono::system_clock::now() - start_time).count() / 1000000000.0) << " seconds" << std::endl;
        }

        //Get centroids after map has been updated
        findObstacleCentroids();

        updating_voronoi = false;
        num_nodes = adj_list.size();

        return true;
    }

    std::vector<std::vector<int>> voronoi_path::getAdjList()
    {
        return adj_list;
    }

    void voronoi_path::printEdges()
    {
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
    }

    std::string voronoi_path::hash(const double &x, const double &y)
    {
        double factor = 1.0 / hash_resolution;
        std::string ret;

        //!!!!!!! Inaccurate results if (float) cast is removed. !!!!!!
        //Suspect it's inconsistent accuracy difference, causing nodes to become disconnected from small rounding errors
        //when converting nodes to hash key-pairs in edgesToGraph() method

        //Number of nodes with (float) cast is usually lesser than without. Indicating disconnected nodes even in close proximity
        std::string x_string = std::to_string((int)((float)x * factor));
        std::string y_string = std::to_string((int)((float)y * factor));
        while (x_string.length() < hash_length)
            x_string.insert(0, "0");

        while (y_string.length() < hash_length)
            y_string.insert(0, "0");

        ret = x_string + y_string;

        return ret;
    }

    std::vector<double> voronoi_path::dehash(const std::string &str)
    {
        std::vector<double> ret(2);
        ret[0] = std::stof(str.substr(0, hash_length)) / 10.0;
        ret[1] = std::stof(str.substr(hash_length, hash_length)) / 10.0;

        return ret;
    }

    std::vector<std::vector<GraphNode>> voronoi_path::getPath(const GraphNode &start, const GraphNode &end, const int &num_paths)
    {
        //Block until voronoi is no longer being updated. Prevents issue where planning is done using an empty adjacency list
        while (true)
        {
            if (!updating_voronoi)
                break;
        }

        is_planning = true;
        auto start_time = std::chrono::system_clock::now();
        std::vector<std::vector<GraphNode>> path;

        int start_node, end_node;
        if (!getNearestNode(start, end, start_node, end_node))
        {
            is_planning = false;
            return path;
        }

        std::vector<int> shortest_path;
        double cost;
        auto shortest_time = std::chrono::system_clock::now();
        if (findShortestPath(start_node, end_node, shortest_path, cost))
        {

            if (print_timings)
                std::cout << "Time taken to find shortest path: " << ((std::chrono::system_clock::now() - shortest_time).count() / 1000000000.0) << "s\n";

            std::vector<std::vector<int>> all_paths;
            auto kth_time = std::chrono::system_clock::now();
            //Get next shortest path
            if (num_paths >= 1)
            {
                std::cout << "Finding alternate paths\n";
                try
                {
                    kthShortestPaths(start_node, end_node, shortest_path, all_paths, num_paths - 1);
                }
                catch (const std::exception &e)
                {
                    std::cout << "Exception while finding alternate paths, failed to find alternate paths\n";
                    std::cout << e.what() << std::endl;
                }
            }

            if (print_timings)
                std::cout << "Time taken to find alternate paths: " << ((std::chrono::system_clock::now() - kth_time).count() / 1000000000.0) << "s\n";

            //Copy all_paths into new container which include start and end
            std::vector<std::vector<GraphNode>> all_path_nodes;
            all_path_nodes.reserve(all_paths.size());

            for (int i = 0; i < all_paths.size(); ++i)
            {
                int curr_path_size = all_paths[i].size();

                all_path_nodes.push_back(std::vector<GraphNode>{start});
                all_path_nodes[i].reserve(curr_path_size + 2);

                for (int j = 0; j < curr_path_size; ++j)
                {
                    all_path_nodes[i].push_back(GraphNode(node_inf[all_paths[i][j]].x, node_inf[all_paths[i][j]].y));
                }

                all_path_nodes[i].push_back(end);
            }

            //Bezier interpolation
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
                            sub_nodes.push_back(prev_2_nodes[1] + dir * extra_point_distance * map.resolution);

                            //Do not insert if collision occurs when extra point is added
                            if (edgeCollides(sub_nodes[sub_nodes.size() - 2], sub_nodes.back()))
                                sub_nodes.pop_back();

                            prev_2_nodes.clear();
                        }
                    }

                    //If this node to the next node does not collide
                    if (!edgeCollides(all_path_nodes[j][i - 1], all_path_nodes[j][i]) &&
                        !edgeCollides(sub_nodes[0], all_path_nodes[j][i]) &&
                        sub_nodes.size() < bezier_max_n)
                    {
                        sub_nodes.push_back(all_path_nodes[j][i]);
                    }

                    //Collision happened or limit reached, find sub path with current sub nodes
                    else
                    {
                        //Retrace back i value to prevent skipping a node
                        --i;

                        //Calculate the bezier subsection
                        std::vector<GraphNode> temp_bezier = bezierSubsection(sub_nodes);
                        bezier_path.insert(bezier_path.end(), temp_bezier.begin(), temp_bezier.end());
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
            }

            //No bezier interpolation
            // path = all_path_nodes;

            if (print_timings)
                std::cout << "Time taken to find all paths, including time to find nearest node: " << ((std::chrono::system_clock::now() - start_time).count() / 1000000000.0) << "s\n";
        }

        else
            std::cout << "Path could not be found" << std::endl;

        is_planning = false;
        return path;
    }

    bool voronoi_path::getNearestNode(const GraphNode &start, const GraphNode &end, int &start_node, int &end_node)
    {
        auto start_time = std::chrono::system_clock::now();
        //TODO: Should not only check nearest nodes. Should allow nearest position to be on an edge
        //Find node nearest to starting point and ending point
        double start_end_vec[2] = {end.x - start.x, end.y - start.y};
        double start_next_vec[2];

        double min_start_dist = std::numeric_limits<double>::infinity();
        double min_rev_start_dist = std::numeric_limits<double>::infinity();
        double min_end_dist = std::numeric_limits<double>::infinity();
        int reverse_start = -1;
        start_node = -1;
        end_node = -1;

        double ang, temp_end_dist, temp_start_dist;
        GraphNode curr;

        for (int i = 0; i < num_nodes; ++i)
        {
            curr.x = node_inf[i].x;
            curr.y = node_inf[i].y;

            start_next_vec[0] = curr.x - start.x;
            start_next_vec[1] = curr.y - start.y;

            temp_start_dist = pow(curr.x - start.x, 2) + pow(curr.y - start.y, 2);
            temp_end_dist = pow(curr.x - end.x, 2) + pow(curr.y - end.y, 2);

            //If potential starting node brings robot towards end goal
            if (temp_start_dist < min_start_dist)
            {
                ang = fabs(vectorAngle(start_end_vec, start_next_vec));
                if (ang < M_PI / 2.0)
                {
                    if (!edgeCollides(start, curr))
                    {
                        min_start_dist = temp_start_dist;
                        start_node = i;
                    }
                }

                //TODO: Should consider reverse start here
            }

            //Else if potential starting node requires robot to go away from end goal
            else if (temp_start_dist < min_rev_start_dist)
            {
                ang = fabs(vectorAngle(start_end_vec, start_next_vec));
                if (ang >= M_PI / 2.0)
                {
                    if (!edgeCollides(start, curr))
                    {
                        min_rev_start_dist = temp_start_dist;
                        reverse_start = i;
                    }
                }
            }

            if (temp_end_dist < min_end_dist)
            {
                if (!edgeCollides(end, curr))
                {
                    min_end_dist = temp_end_dist;
                    end_node = i;
                }
            }
        }

        //Relax criterion on requiring forward start if forward start nearest node is not found
        if (start_node == -1)
            start_node = reverse_start;

        //Failed to find start/end even after relaxation
        if (start_node == -1 || end_node == -1)
        {
            std::cout << "Failed to find nearest starting or ending node" << std::endl;
            return false;
        }

        if (print_timings)
            std::cout << "Time taken to find nearest node: " << ((std::chrono::system_clock::now() - start_time).count() / 1000000000.0) << "s\n";
        return true;
    }

    std::complex<double> voronoi_path::fNaught(const std::complex<double> &z, const int &n)
    {
        //a - b = 0
        //a + b = n - 1
        //TODO: Implement check that this is a valid combination of a and b
        double a = (n - 1) / 2.0;
        double b = a;
        return std::pow((z - BL), a) + std::pow((z - TR), b);
    }

    std::complex<double> voronoi_path::calcHomotopyClass(const std::vector<int> &path_)
    {
        std::vector<std::complex<double>> path;
        path.reserve(path_.size());

        //Convert path to complex path, emplace back causes crashing
        for (auto node : path_)
            path.emplace_back(node_inf[node].x, node_inf[node].y);

        std::complex<double> path_sum(0, 0);
        //Go through each edge of the path
        for (int i = 1; i < path.size(); ++i)
        {
            std::complex<double> edge_sum(0, 0);
            //Each edge must iterate through all obstacles
            for (auto obs : centers)
            {
                //Calculate Al value, initialize with 1,1
                std::complex<double> al_denom_prod(1, 1);
                for (auto excl_obs : centers)
                {
                    if (excl_obs == obs)
                        continue;

                    al_denom_prod *= (obs - excl_obs);
                }
                std::complex<double> al = fNaught(obs, centers.size()) / al_denom_prod;

                double real_part = std::log(std::abs(path[i] - obs)) - std::log(std::abs(path[i - 1] - obs));
                double im_part = std::arg(path[i] - obs) - std::arg(path[i - 1] - obs);

                //Get smallest angle
                while (im_part > M_PI)
                    im_part -= 2 * M_PI;

                while (im_part < -M_PI)
                    im_part += 2 * M_PI;

                edge_sum += (std::complex<double>(real_part, im_part) * al);
            }
            //Add this edge's sum to the path sum
            path_sum += edge_sum;
        }

        return path_sum;
    }

    //TODO: There is issue with path reversing then going towards first shortest path
    //TODO: Occasionally crashes
    //TODO: Check efficiency
    bool voronoi_path::kthShortestPaths(const int &start_node, const int &end_node, const std::vector<int> &shortestPath, std::vector<std::vector<int>> &all_paths, const int &num_paths)
    {
        double adjacency_cum_time = 0;
        double copy_root_cum_time = 0;
        double disconnect_edge_cum_time = 0;
        double remove_nodes_cum_time = 0;
        double spur_path_cum_time = 0;
        double copy_kth_cum_time = 0;
        double get_total_cost_time = 0;

        all_paths.reserve(num_paths + 1);

        try
        {
            if (num_paths == 0)
            {
                all_paths.push_back(shortestPath);
                return true;
            }

            std::vector<std::vector<int>> kthPaths;
            kthPaths.reserve(num_paths + 1);
            kthPaths.push_back(shortestPath);
            std::vector<std::vector<int>> adj_list_backup(adj_list);
            std::vector<int> adj_list_removed_edges;
            std::vector<int> adj_list_modified_ind;
            int last_root_ind = 0;

            std::vector<std::vector<int>> potentialKth;
            std::vector<double> cost_vec;

            for (int k = 1; k <= num_paths; ++k)
            {
                //Break if cannot find the previous k path, could not find all NUM_PATHS
                if (k - 1 == kthPaths.size())
                    break;

                cost_vec.clear();
                potentialKth.clear();

                last_root_ind = 0;

                //Spur node is ith node
                for (int i = 0; i < kthPaths[k - 1].size() - 2; ++i)
                {
                    //Restore/copy adjacency list
                    auto restore_start = std::chrono::system_clock::now();
                    for (int z = 0; z < adj_list_removed_edges.size(); ++z)
                    {
                        adj_list[adj_list_removed_edges[z]] = adj_list_backup[adj_list_removed_edges[z]];
                    }

                    adj_list_removed_edges.clear();
                    adj_list_removed_edges.shrink_to_fit();
                    adjacency_cum_time += (std::chrono::system_clock::now() - restore_start).count() / 1000000000.0;

                    //Spur node is from start to 2nd last node of path, inclusive
                    int spurNode = kthPaths[k - 1][i];
                    std::vector<int> rootPath(i);

                    //Copy root path into container. Root path is path before spur node, containing the path from start onwards
                    //Root path size might be 0 if spurNode is the starting node
                    auto copy_root_time = std::chrono::system_clock::now();
                    std::copy(kthPaths[k - 1].begin(), kthPaths[k - 1].begin() + i, rootPath.begin());
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
                        if (equal_count == rootPath.size())
                        {
                            //Remove edge from spurNode to spur_next
                            int spur_next = kthPaths[k - 1][i + 1];
                            auto erase_it = std::find(adj_list[spurNode].begin(), adj_list[spurNode].end(), spur_next);

                            //Edge already removed
                            if (erase_it == adj_list[spurNode].end())
                                continue;

                            adj_list[spurNode][erase_it - adj_list[spurNode].begin()] = -1;
                            adj_list_removed_edges.push_back(spurNode);

                            //Remove edge from spur_next to spurNode
                            erase_it = std::find(adj_list[spur_next].begin(), adj_list[spur_next].end(), spurNode);

                            //Edge already removed
                            if (erase_it == adj_list[spurNode].end())
                                continue;

                            adj_list[spur_next][erase_it - adj_list[spur_next].begin()] = -1;
                            adj_list_removed_edges.push_back(spur_next);

                            break;
                        }
                    }
                    disconnect_edge_cum_time += (std::chrono::system_clock::now() - disconnect_edges_time).count() / 1000000000.0;

                    //Remove all nodes of root path from graph except spur node and start node
                    auto node_start_time = std::chrono::system_clock::now();

                    int node = 0;
                    for (int node_ind = last_root_ind; node_ind < rootPath.size(); ++node_ind)
                    // for (auto node : rootPath)
                    {
                        node = rootPath[node_ind];
                        if (node == spurNode || node == start_node)
                            continue;

                        for (int del_ind = 0; del_ind < adj_list[node].size(); ++del_ind)
                        {
                            //Remove connection from point-er side
                            int pointed_to = adj_list[node][del_ind];

                            //Edge is already deleted
                            if (pointed_to == -1)
                                continue;

                            adj_list[node][del_ind] = -1;
                            adj_list_modified_ind.push_back(node);

                            //Remove connection from point-ed side
                            auto erase_it = std::find(adj_list[pointed_to].begin(), adj_list[pointed_to].end(), node);
                            if (erase_it != adj_list[pointed_to].end())
                            {
                                adj_list[pointed_to][erase_it - adj_list[pointed_to].begin()] = -1;
                                adj_list_modified_ind.push_back(pointed_to);
                            }

                            last_root_ind = node_ind;
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

                        if (rootPath.size())
                            total_path.insert(total_path.begin(), rootPath.begin(), rootPath.end() - 1);
                        total_path.insert(total_path.end(), spur_path.begin(), spur_path.end());

                        //Check if the path just generated is unique in the potential k paths
                        bool path_is_unique = true;

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
                                    if (node_pot < kthPaths[check_path].size() && node_pot < total_path.size())
                                        if (kthPaths[check_path][node_pot] == total_path[node_pot])
                                            kth_equal++;
                                }

                                //Compare with potentialKths
                                if (check_path < potentialKth.size())
                                {
                                    if (node_pot < potentialKth[check_path].size() && node_pot < total_path.size())
                                        if (potentialKth[check_path][node_pot] == total_path[node_pot])
                                            pot_equal++;
                                }
                            }

                            if (pot_equal == total_path.size() || kth_equal == total_path.size())
                                path_is_unique = false;
                        }

                        //add unique path to potentials
                        auto get_total_start = std::chrono::system_clock::now();
                        if (path_is_unique)
                        {
                            //Get cost of total path
                            double total_cost;
                            for (int int_node = 0; int_node < total_path.size() - 1; ++int_node)
                            {
                                total_cost += euclideanDist(node_inf[int_node], node_inf[int_node + 1]);
                            }

                            cost_vec.push_back(total_cost);
                            potentialKth.push_back(total_path);
                        }
                        get_total_cost_time += (std::chrono::system_clock::now() - get_total_start).count() / 1000000000.0;
                    }
                }

                auto copy_kth = std::chrono::system_clock::now();
                //Find minimum cost path
                double min_cost = std::numeric_limits<double>::infinity();
                int copy_index = 0;
                for (int min_ind = 0; min_ind < cost_vec.size(); ++min_ind)
                {
                    if (cost_vec[min_ind] < min_cost)
                    {

                        //Calculate homotopy class for all previously generated paths
                        std::vector<std::complex<double>> homotopy_classes;
                        for (auto homotopy_paths : kthPaths)
                        {
                            homotopy_classes.emplace_back(calcHomotopyClass(homotopy_paths));
                        }

                        //Get the current potential path's homotopy class
                        std::complex<double> curr_h_class = calcHomotopyClass(potentialKth[min_ind]);

                        //Check that the path is in unique homotopy class compared to previous kthPaths
                        //Iterate through all path's homotopy class
                        for (auto h_class : homotopy_classes)
                        {
                            //This homotopy class path does not exist yet, assign min_cost and copy_index
                            if (std::abs(curr_h_class - h_class) > h_class_threshold)
                            {

                                min_cost = cost_vec[min_ind];
                                copy_index = min_ind;
                            }
                        }
                    }
                }

                //Copy lowest cost path into kthPaths, if not already inside
                bool path_unique = true;
                for (auto paths : kthPaths)
                {
                    int equal_count = 0;
                    for (int path_nodes = 0; path_nodes < paths.size(); ++path_nodes)
                    {
                        try
                        {
                            if (paths[path_nodes] == potentialKth[copy_index][path_nodes])
                                equal_count++;
                        }

                        catch (const std::exception &e)
                        {
                            break;
                        }
                    }

                    if (equal_count == potentialKth[copy_index].size())
                    {
                        path_unique = false;
                        break;
                    }
                }

                if (path_unique)
                    kthPaths.push_back(potentialKth[copy_index]);
                copy_kth_cum_time += (std::chrono::system_clock::now() - copy_kth).count() / 1000000000.0;

                //Reset entire adj_list when k path changes
                for (int i = 0; i < adj_list_modified_ind.size(); ++i)
                {
                    adj_list[adj_list_modified_ind[i]] = adj_list_backup[adj_list_modified_ind[i]];
                }
                adj_list_modified_ind.clear();
                adj_list_modified_ind.shrink_to_fit();

                if (potentialKth.size() == 0)
                    break;
            }
            all_paths.insert(all_paths.begin(), kthPaths.begin(), kthPaths.end());
        }

        catch (const std::exception &e)
        {
            std::cout << "Exception while finding alternate paths, failed to find alternate paths\n";
            std::cout << e.what() << std::endl;
            all_paths.push_back(shortestPath);
            return false;
        }

        if (print_timings)
        {
            std::cout << "Cumulative adjacency list restore time " << adjacency_cum_time << std::endl;
            std::cout << "Cumulative copy root path time " << copy_root_cum_time << std::endl;
            std::cout << "Cumulative disconnect edges time " << disconnect_edge_cum_time << std::endl;
            std::cout << "Cumulative remove nodes of root path time " << remove_nodes_cum_time << std::endl;
            std::cout << "Cumulative find spur path time " << spur_path_cum_time << std::endl;
            std::cout << "Cumulative copy kth path time " << copy_kth_cum_time << std::endl;
            std::cout << "Cumulative get total cost time " << get_total_cost_time << std::endl;
        }

        if (num_paths == all_paths.size() - 1)
            return true;

        else
            return false;
    }

    bool voronoi_path::findShortestPath(const int &start_node, const int &end_node, std::vector<int> &path, double &cost)
    {
        auto start_time = std::chrono::system_clock::now();
        std::vector<std::pair<int, NodeInfo>> closed_list;
        std::vector<std::pair<int, NodeInfo>> open_list;
        std::vector<bool> nodes_closed_bool(num_nodes, false);
        std::vector<int> nodes_prev(num_nodes, -1);

        NodeInfo start_info;
        start_info.prevNode = -1;
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

        while (curr_node != end_node)
        {
            curr_node = open_list[min_ind].first;
            curr_node_location = node_inf[curr_node];
            //Get info for current node
            auto open_start = std::chrono::system_clock::now();
            curr_node_info = std::find_if(open_list.begin(), open_list.end(),
                                          [&curr_node](const std::pair<int, NodeInfo> &in) {
                                              return in.first == curr_node;
                                          })
                                 ->second;
            open_list_time += (std::chrono::system_clock::now() - open_start).count() / 1000000000.0;

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

                //Find next_node in open_list
                auto open_start = std::chrono::system_clock::now();
                auto it = std::find_if(open_list.begin(), open_list.end(),
                                       [&next_node](const std::pair<int, NodeInfo> &in) {
                                           return in.first == next_node;
                                       });
                open_list_time += (std::chrono::system_clock::now() - open_start).count() / 1000000000.0;

                next_node_location = node_inf[next_node];
                double here_to_next_dist = euclideanDist(curr_node_location, next_node_location) + curr_node_info.cost_upto_here;

                //If node is not in open list yet
                if (it == open_list.end())
                {
                    NodeInfo new_node;

                    new_node.prevNode = curr_node;
                    new_node.cost_upto_here = here_to_next_dist;
                    new_node.cost_to_goal = euclideanDist(end_node_location, next_node_location);
                    new_node.updateCost();

                    nodes_prev[next_node] = curr_node;

                    open_list.emplace_back(std::make_pair(next_node, new_node));
                }

                else
                {
                    //Update node's nearest distance to reach here, if the new cost is lower
                    if (here_to_next_dist < it->second.cost_upto_here)
                    {
                        it->second.cost_upto_here = here_to_next_dist;
                        it->second.prevNode = curr_node;
                        it->second.updateCost();

                        nodes_prev[next_node] = curr_node;
                    }
                }
            }

            //Remove curr_node from open list after all adjacent nodes have been put into open list
            open_list.erase(std::find_if(open_list.begin(), open_list.end(),
                                         [&curr_node](const std::pair<int, NodeInfo> &in) { return in.first == curr_node; }));

            //Then put into closed list
            closed_list.emplace_back(std::make_pair(curr_node, curr_node_info));
            nodes_closed_bool[curr_node] = true;

            // Get index of minimum total cost in open list
            if (open_list.size())
            {
                double min_val = std::numeric_limits<double>::infinity();

                for (int i = 0; i < open_list.size(); ++i)
                {
                    double curr_min = open_list[i].second.total_cost;
                    if (curr_min < min_val)
                    {
                        min_val = curr_min;
                        min_ind = i;
                    }
                }
            }

            else
                break;
        }
        find_path_time += (std::chrono::system_clock::now() - start_time).count() / 1000000000.0;

        auto copy_path_start = std::chrono::system_clock::now();
        //Put nodes into path
        std::vector<int> temp_path;
        int path_current = end_node;
        temp_path.push_back(path_current);

        while (path_current != start_node)
        {
            path_current = nodes_prev[path_current];

            //If previous node does not exist;
            if (path_current == -1)
                return false;

            temp_path.push_back(path_current);
        }

        //Path found
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
                int pixel = floor(edge_vector[i].pos[j].x) + floor(edge_vector[i].pos[j].y) * map.width;

                //If vertex pixel in map is not free, remove this edge
                if (map.data[pixel] > collision_threshold)
                {
                    delete_indices.push_back(i);
                    break;
                }
            }
        }

        if (delete_indices.size() != 0)
        {
            std::vector<jcv_edge> remaining_edges;
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
            jcv_edge curr_edge = edge_vector[i];

            GraphNode start(curr_edge.pos[0].x, curr_edge.pos[0].y);
            GraphNode end(curr_edge.pos[1].x, curr_edge.pos[1].y);

            if (edgeCollides(start, end))
                delete_indices.push_back(i);
        }

        if (delete_indices.size() != 0)
        {
            std::vector<jcv_edge> remaining_edges;
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

            pixel = int(curr_x) + int(curr_y) * map.width;
            if (map.data[pixel] > collision_threshold)
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

        //Delete points that are too near
        auto it = points.begin() + 1;
        double curr_x = -1;
        double curr_y = -1;
        double pixel_threshold = min_node_sep_sq * map.resolution;
        while (it < points.end())
        {
            if (curr_x < 0 && curr_y < 0)
            {
                curr_x = it->x;
                curr_y = it->y;
                continue;
            }

            double dist = pow(it->x - curr_x, 2) + pow(it->y - curr_y, 2);
            if (dist < pixel_threshold && (it != points.end() - 1))
            {
                it = points.erase(it);
                curr_x = -1;
                curr_y = -1;
            }

            ++it;
        }

        int n = points.size() - 1;
        std::vector<int> combos(n + 1);
        std::vector<GraphNode> bezier_path;

        //Calculate all required nCk
        for (int i = 0; i < n + 1; ++i)
        {
            combos[i] = binomialCoeff(n, i);
        }

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