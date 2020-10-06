#include "shared_voronoi_global_planner.h"
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>
#include <thread>
#include <future>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <limits>
#include <geometry_msgs/Point.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

PLUGINLIB_EXPORT_CLASS(shared_voronoi_global_planner::SharedVoronoiGlobalPlanner, nav_core::BaseGlobalPlanner)

namespace shared_voronoi_global_planner
{
    SharedVoronoiGlobalPlanner::SharedVoronoiGlobalPlanner()
        : nh("~" + std::string("SharedVoronoiGlobalPlanner"))
    {
    }

    SharedVoronoiGlobalPlanner::SharedVoronoiGlobalPlanner(std::string name, costmap_2d::Costmap2DROS *costmap_ros)
        : nh("~" + std::string("SharedVoronoiGlobalPlanner"))
    {
    }

    void SharedVoronoiGlobalPlanner::updateVoronoiCB(const ros::WallTimerEvent &e)
    {
        if (map.data.empty())
        {
            ROS_WARN("Map is still empty, skipping update of voronoi diagram");
            return;
        }

        voronoi_path.mapToGraph(map);

        if(print_edges)
            voronoi_path.printEdges();
    }

    void SharedVoronoiGlobalPlanner::updateVoronoiMapCB(const ros::WallTimerEvent &e)
    {
        auto update_start = std::chrono::system_clock::now();
        map.height = merged_costmap.info.height;
        map.width = merged_costmap.info.width;
        map.frame_id = merged_costmap.header.frame_id;
        map.resolution = merged_costmap.info.resolution;
        map.origin.position.x = merged_costmap.info.origin.position.x;
        map.origin.position.y = merged_costmap.info.origin.position.y;

        map.data.clear();
        map.data.insert(map.data.begin(), merged_costmap.data.begin(), merged_costmap.data.end());
        threadedMapCleanup();
        // std::cout << "Update voronoi map: " << (std::chrono::system_clock::now() - update_start).count()/1000000000.0 << "\n";
    }

    void SharedVoronoiGlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros)
    {
        if (!initialized_)
        {
            //Read parameters
            nh.getParam("occupancy_threshold", occupancy_threshold);
            nh.getParam("update_voronoi_rate", update_voronoi_rate);
            nh.getParam("update_costmap_rate", update_costmap_rate);
            nh.getParam("print_timings", print_timings);
            nh.getParam("hash_length", hash_length);
            nh.getParam("line_check_resolution", line_check_resolution);
            nh.getParam("pixels_to_skip", pixels_to_skip);
            nh.getParam("open_cv_scale", open_cv_scale);
            nh.getParam("h_class_threshold", h_class_threshold);
            nh.getParam("min_node_sep_sq", min_node_sep_sq);
            nh.getParam("extra_point_distance", extra_point_distance);
            nh.getParam("add_local_costmap_corners", add_local_costmap_corners);
            nh.getParam("forward_sim_time", forward_sim_time);
            nh.getParam("num_paths", num_paths);
            nh.getParam("publish_all_path_markers", publish_all_path_markers);
            nh.getParam("user_dir_filter", user_dir_filter);
            nh.getParam("min_edge_length", min_edge_length);
            nh.getParam("joystick_topic", joystick_topic);
            nh.getParam("print_edges", print_edges);

            voronoi_path.h_class_threshold = h_class_threshold;
            voronoi_path.print_timings = print_timings;
            voronoi_path.min_edge_length = min_edge_length;

            //Subscribe and advertise related topics
            global_costmap_sub = nh.subscribe("/move_base/global_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::globalCostmapCB, this);
            global_update_sub = nh.subscribe("/move_base/global_costmap/costmap_updates", 1, &SharedVoronoiGlobalPlanner::globalCostmapUpdateCB, this);
            local_costmap_sub = nh.subscribe("/move_base/local_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::localCostmapCB, this);
            user_vel_sub = nh.subscribe("/test_vel", 1, &SharedVoronoiGlobalPlanner::cmdVelCB, this);

            merged_costmap_pub = nh.advertise<nav_msgs::OccupancyGrid>("merged_costmap", 1);
            global_path_pub = nh.advertise<nav_msgs::Path>("plan", 1);
            all_paths_pub = nh.advertise<visualization_msgs::MarkerArray>("all_paths", 1);
            user_direction_pub = nh.advertise<visualization_msgs::Marker>("user_direction", 1);

            if (publish_centroids)
                centroid_pub = nh.advertise<visualization_msgs::MarkerArray>("centroids", 1);

            //Create timer to update Voronoi diagram
            voronoi_update_timer = nh.createWallTimer(ros::WallDuration(1.0 / update_voronoi_rate), &SharedVoronoiGlobalPlanner::updateVoronoiCB, this);
            map_update_timer = nh.createWallTimer(ros::WallDuration(1.0 / update_costmap_rate), &SharedVoronoiGlobalPlanner::updateVoronoiMapCB, this);

            ROS_INFO("Shared Voronoi Global Planner initialized");
        }

        else
            ROS_INFO("Shared Voronoi Global Planner already initialized, not doing anything");
    }

    bool SharedVoronoiGlobalPlanner::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal, std::vector<geometry_msgs::PoseStamped> &plan)
    {
        //Get start and end points in terms of global costmap pixels
        voronoi_path::GraphNode start_point((start.pose.position.x - merged_costmap.info.origin.position.x) / merged_costmap.info.resolution,
                                            (start.pose.position.y - merged_costmap.info.origin.position.y) / merged_costmap.info.resolution);
        voronoi_path::GraphNode end_point((goal.pose.position.x - merged_costmap.info.origin.position.x) / merged_costmap.info.resolution,
                                          (goal.pose.position.y - merged_costmap.info.origin.position.y) / merged_costmap.info.resolution);

        ros::Rate r(5);
        while (true)
        {
            if (!voronoi_path.isUpdatingVoronoi())
                break;

            ROS_DEBUG("Voronoi diagram is updating, waiting for it to complete before requesting for plan");
            r.sleep();
        }

        //Reset previous path if goal has changed
        if (prev_goal.header.frame_id.empty())
            prev_goal = goal;

        else
        {
            //If goal has changed
            if (prev_goal.pose.position.x != goal.pose.position.x ||
                prev_goal.pose.position.y != goal.pose.position.y ||
                prev_goal.pose.position.z != goal.pose.position.z)
            {
                prev_goal = goal;
                prev_path.clear();
            }
        }

        //Get voronoi paths
        std::vector<std::vector<voronoi_path::GraphNode>> all_paths = voronoi_path.getPath(start_point, end_point, num_paths);

        if (all_paths.size() != num_paths)
            ROS_WARN("Could not find all requested paths. Requested: %d, found: %ld", num_paths, all_paths.size());

        //If paths are found
        if (!all_paths.empty())
        {
            std::vector<std::vector<geometry_msgs::PoseStamped>> all_paths_meters(all_paths.size());
            visualization_msgs::MarkerArray marker_array;

            //Convert node numbers to position on map for path
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = merged_costmap.header.frame_id;
            for (int i = 0; i < all_paths.size(); ++i)
            {
                visualization_msgs::Marker marker;
                if (publish_all_path_markers)
                {
                    //Create marker item to store a line
                    marker.header = header;
                    marker.ns = std::string("Path ") + std::to_string(i);
                    marker.id = i;
                    marker.type = 4;
                    marker.action = 0;
                    marker.scale.x = 0.05;
                    marker.color.r = (255 / all_paths.size() * i) / 255.0;
                    marker.color.g = (255 / all_paths.size() * i) / 255.0;
                    marker.color.b = (255 / all_paths.size() * i) / 255.0;
                    marker.color.a = 1.0;
                    marker.pose.orientation.w = 1.0;
                    marker.lifetime = ros::Duration(1.0);
                }

                //Loop through all the nodes for path i
                for (int j = 0; j < all_paths[i].size(); ++j)
                {
                    geometry_msgs::PoseStamped new_pose;
                    new_pose.header = header;
                    new_pose.pose.position.x = all_paths[i][j].x * merged_costmap.info.resolution + merged_costmap.info.origin.position.x;
                    new_pose.pose.position.y = all_paths[i][j].y * merged_costmap.info.resolution + merged_costmap.info.origin.position.y;
                    new_pose.pose.position.z = 0;

                    new_pose.pose.orientation.w = 1;

                    all_paths_meters[i].push_back(new_pose);

                    if (publish_all_path_markers)
                        marker.points.push_back(new_pose.pose.position);
                }

                //Adjust orientation of start and end positions
                if (!all_paths_meters[i].empty())
                {
                    all_paths_meters[i][0] = start;
                    all_paths_meters[i].back() = goal;
                }

                if (publish_all_path_markers)
                    marker_array.markers.push_back(marker);
            }

            //Publish visualization for all available paths
            if (publish_all_path_markers)
                all_paths_pub.publish(marker_array);

            //Prune previous selected path
            while (!prev_path.empty())
            {
                double dist = sqrt(pow(prev_path[0].pose.position.x - start.pose.position.x, 2) + pow(prev_path[0].pose.position.y - start.pose.position.y, 2));
                if (dist < 0.3)
                    prev_path.erase(prev_path.begin());

                else
                    break;
            }

            //Add previously selected path into selections
            if(!prev_path.empty())
                all_paths_meters.push_back(prev_path);

            //Select the path most similar to user commanded velocity path
            int preferred_path = 0;
            double dist = pow(start.pose.position.x - goal.pose.position.x, 2) + pow(start.pose.position.y - goal.pose.position.y, 2);
            if ((cmd_vel.linear.x != 0.0 || cmd_vel.angular.z != 0.0) && dist > pow(near_goal_threshold, 2))
                preferred_path = getMatchedPath(start, all_paths_meters);

            //Set selected plan
            if (!all_paths_meters[preferred_path].empty())
            {
                plan = all_paths_meters[preferred_path];
                prev_path = plan;
            }

            //Publish selected plan for visualization
            nav_msgs::Path viz_path;
            viz_path.header.stamp = ros::Time::now();
            viz_path.header.frame_id = merged_costmap.header.frame_id;
            viz_path.poses = all_paths_meters[preferred_path];
            global_path_pub.publish(viz_path);

            return true;
        }

        else
            return false;
    }

    int SharedVoronoiGlobalPlanner::getMatchedPath(const geometry_msgs::PoseStamped &curr_pose,
                                                   const std::vector<std::vector<geometry_msgs::PoseStamped>> &plans_)
    {
        auto get_matched_start = std::chrono::system_clock::now();
        //Forward simulate commanded velocity
        double lin_x = 0.2;
        double time_interval = fabs(forward_sim_resolution / lin_x);

        std::vector<std::pair<double, double>> user_path;
        user_path.reserve(int(forward_sim_time / time_interval));

        //Create marker item to store a line
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = merged_costmap.header.frame_id;
        marker.ns = "User direction";
        marker.id = 0;
        marker.type = 4;
        marker.action = 0;
        marker.scale.x = 0.05;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        marker.pose.orientation.w = 1.0;
        marker.lifetime = ros::Duration(1.0);

        double x = curr_pose.pose.position.x;
        double y = curr_pose.pose.position.y;
        // double theta = tf::getYaw(curr_pose.pose.orientation) + atan2(cmd_vel.angular.z, cmd_vel.linear.x);

        double new_local_dir = user_dir_filter * atan2(cmd_vel.angular.z, cmd_vel.linear.x) + (1 - user_dir_filter) * prev_local_dir;
        double theta = tf::getYaw(curr_pose.pose.orientation) + new_local_dir;
        prev_local_dir = new_local_dir;

        user_path.emplace_back(x, y);

        double curr_time = 0.0;
        while (curr_time <= forward_sim_time + 0.01)
        {
            if (curr_time > forward_sim_time)
                curr_time = forward_sim_time;

            x += lin_x * cos(theta) * time_interval;
            y += lin_x * sin(theta) * time_interval;

            geometry_msgs::Point point;
            point.x = x;
            point.y = y;
            point.z = 0;

            user_path.emplace_back(x, y);
            marker.points.push_back(point);

            curr_time += time_interval;
        }

        user_direction_pub.publish(marker);

        double total_distance = sqrt(pow(user_path.back().first - user_path[0].first, 2) +
                                     pow(user_path.back().second - user_path[0].second, 2));

        std::vector<double> ang_diff_sq(plans_.size(), 0);
        double max_s = total_distance;

        double vec1[] = {user_path.back().first - user_path[0].first,
                         user_path.back().second - user_path[0].second};

        //Iterate over all paths
        for (int i = 0; i < plans_.size(); ++i)
        {
            double curr_s_along_path = 0;

            //Iterate over all points in path
            for (int j = 1; j < plans_[i].size(); ++j)
            {
                //Calculate incremental increase in displacement along path
                double ds = sqrt(pow(plans_[i][j - 1].pose.position.x - plans_[i][j].pose.position.x, 2) +
                                 pow(plans_[i][j - 1].pose.position.y - plans_[i][j].pose.position.y, 2));
                curr_s_along_path += ds;

                if (curr_s_along_path > max_s)
                    break;

                double vec2[] = {plans_[i][j].pose.position.x - plans_[i][j - 1].pose.position.x,
                                 plans_[i][j].pose.position.y - plans_[i][j - 1].pose.position.y};

                ang_diff_sq[i] += pow(vectorAngle(vec1, vec2), 2);
            }
        }

        // std::cout << "Get matched path: " << (std::chrono::system_clock::now() - get_matched_start).count()/1000000000.0 << "\n";
        return std::min_element(ang_diff_sq.begin(), ang_diff_sq.end()) - ang_diff_sq.begin();
    }

    double SharedVoronoiGlobalPlanner::vectorAngle(const double vec1[2], const double vec2[2])
    {
        double dot = vec1[0] * vec2[0] + vec1[1] * vec2[1];
        double det = vec1[0] * vec2[1] - vec1[1] * vec2[0];
        return std::atan2(det, dot);
    }

    void SharedVoronoiGlobalPlanner::localCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        auto local_start = std::chrono::system_clock::now();
        local_costmap = *msg;

        //Merge costmaps if global map is not empty
        if (!local_costmap.data.empty())
        {
            //Get origin of local_costmap wrt to origin of global_costmap
            double rel_local_x = -merged_costmap.info.origin.position.x + local_costmap.info.origin.position.x;
            double rel_local_y = -merged_costmap.info.origin.position.y + local_costmap.info.origin.position.y;

            //Costmap is rotated ccw 90deg in rviz
            //Convert distance to pixels in global costmap resolution
            int x_pixel_offset = rel_local_x / merged_costmap.info.resolution;
            int y_pixel_offset = rel_local_y / merged_costmap.info.resolution;

            std::vector<voronoi_path::GraphNode> local_vertices;
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset, y_pixel_offset));
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset + local_costmap.info.width, y_pixel_offset));
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset + local_costmap.info.width, y_pixel_offset + local_costmap.info.height));
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset, y_pixel_offset + local_costmap.info.height));

            if (add_local_costmap_corners)
                voronoi_path.setLocalVertices(local_vertices);

            //Restore modified global costmap pixels to old value in previous loop
            for (int i = 0; i < map_pixels_backup.size(); ++i)
            {
                merged_costmap.data[map_pixels_backup[i].first] = map_pixels_backup[i].second;
            }
            map_pixels_backup.clear();

            for (int i = 0; i < local_costmap.data.size(); ++i)
            {
                int local_data = local_costmap.data[i];

                if (local_data >= occupancy_threshold)
                {
                    int global_curr_x = i % local_costmap.info.width + x_pixel_offset;
                    int global_curr_y = i / local_costmap.info.width + y_pixel_offset;
                    // map.data[global_curr_y * merged_costmap.info.width + global_curr_x] = local_data;
                    map_pixels_backup.emplace_back(global_curr_y * merged_costmap.info.width + global_curr_x,
                                                   merged_costmap.data[global_curr_y * merged_costmap.info.width + global_curr_x]);
                    merged_costmap.data[global_curr_y * merged_costmap.info.width + global_curr_x] = local_data;
                }
            }
        }

        // std::cout << "Local cb: " << (std::chrono::system_clock::now() - local_start).count()/1000000000.0 << "\n";
        merged_costmap_pub.publish(merged_costmap);
    }

    void SharedVoronoiGlobalPlanner::globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        merged_costmap = *msg;
    }

    void SharedVoronoiGlobalPlanner::globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg)
    {
        auto global_start = std::chrono::system_clock::now();
        merged_costmap.data = msg->data;

        //Copy back local costmap pixels to prevent oscillation of existence in local obstacles
        if (!merged_costmap.data.empty())
        {
            if (!local_costmap.data.empty())
            {
                const auto costmap_ptr = boost::make_shared<nav_msgs::OccupancyGrid>(local_costmap);
                localCostmapCB(costmap_ptr);
            }
        }

        // std::cout << "Global cb: " << (std::chrono::system_clock::now() - global_start).count()/1000000000.0 << "\n";
    }

    void SharedVoronoiGlobalPlanner::cmdVelCB(const geometry_msgs::Twist::ConstPtr &msg)
    {
        cmd_vel = *msg;
    }

    void SharedVoronoiGlobalPlanner::threadedMapCleanup()
    {
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<std::vector<int>>> future_vector;
        future_vector.reserve(num_threads);

        int size = map.data.size();
        int num_pixels = floor(size / num_threads);
        int start_pixel = 0;

        for (int i = 0; i < num_threads; ++i)
        {
            start_pixel = i * num_pixels;
            if (i == num_threads - 1)
                num_pixels = size - num_pixels * (int)(num_threads - 1);

            future_vector.emplace_back(
                std::async(
                    std::launch::async, [start_pixel, num_pixels](const voronoi_path::Map &map) {
                        std::vector<int> edited_map;
                        edited_map.reserve(num_pixels);

                        for (int i = start_pixel; i < start_pixel + num_pixels; i++)
                        {
                            int cur_data = map.data[i];
                            if (cur_data == -1)
                                edited_map.push_back(0);

                            else
                                edited_map.push_back(cur_data);
                        }

                        return edited_map;
                    },
                    std::ref(map)));
        }

        map.data.clear();
        for (int i = 0; i < future_vector.size(); i++)
        {
            future_vector[i].wait();
            std::vector<int> temp_vec = future_vector[i].get();

            map.data.insert(map.data.end(), make_move_iterator(temp_vec.begin()), make_move_iterator(temp_vec.end()));
        }
    }
} // namespace shared_voronoi_global_planner