#include "shared_voronoi_global_planner.h"
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>
// #include <thread>
// #include <future>
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

        voronoi_path.mapToGraph(&map);

        std::vector<voronoi_path::GraphNode> centers;
        voronoi_path.getObstacleCentroids(centers);

        //Publish visualization marker for use in rviz
        if(visualize_edges)
        {
            std::vector<voronoi_path::GraphNode> nodes;
            std::vector<voronoi_path::GraphNode> lonely_nodes;
            voronoi_path.getEdges(nodes);
            voronoi_path.getDisconnectedNodes(lonely_nodes);

            //Markers for voronoi edges
            visualization_msgs::MarkerArray marker_array;
            visualization_msgs::Marker marker;
            marker.header.stamp = ros::Time::now();
            marker.header.frame_id = map.frame_id;
            marker.id = 0;
            marker.ns = "Voronoi Edges";
            marker.type = 5;
            marker.action = 0;
            marker.scale.x = 0.01;
            marker.color.a = 1.0;
            marker.color.b = 1.0;
            marker.lifetime = ros::Duration(0.0);
            marker.pose.orientation.w = 1.0;
            marker.points.reserve(nodes.size());

            for(const auto& node : nodes)
            {
                geometry_msgs::Point temp_point;
                temp_point.x = node.x * static_cast<double>(map.resolution) + map.origin.position.x;
                temp_point.y = node.y * static_cast<double>(map.resolution) + map.origin.position.y;

                if(node.x > 0 && node.x < 0.01 && node.y > 0 && node.y < 0.01)
                    break;

                marker.points.push_back(std::move(temp_point));
            }

            //Markers for voronoi nodes that are only connected on one side
            visualization_msgs::Marker marker_lonely;
            marker_lonely.header.stamp = ros::Time::now();
            marker_lonely.header.frame_id = map.frame_id;
            marker_lonely.id = 1;
            marker_lonely.ns = "Lonely Nodes";
            marker_lonely.type = 8;
            marker_lonely.action = 0;
            marker_lonely.scale.x = 0.1;
            marker_lonely.scale.y = 0.1;
            marker_lonely.color.a = 1.0;
            marker_lonely.color.r = 1.0;
            marker_lonely.lifetime = ros::Duration(0.0);
            marker_lonely.pose.orientation.w = 1.0;
            marker_lonely.points.reserve(lonely_nodes.size());

            for(const auto& node : lonely_nodes)
            {
                geometry_msgs::Point temp_point;
                temp_point.x = node.x * static_cast<double>(map.resolution) + map.origin.position.x;
                temp_point.y = node.y * static_cast<double>(map.resolution) + map.origin.position.y;

                marker_lonely.points.push_back(std::move(temp_point));
            }

            //Markers for centroids of obstacles
            visualization_msgs::Marker marker_obstacles;
            marker_obstacles.header.stamp = ros::Time::now();
            marker_obstacles.header.frame_id = map.frame_id;
            marker_obstacles.id = 2;
            marker_obstacles.ns = "Obstacle Centroids";
            marker_obstacles.type = 8;
            marker_obstacles.action = 0;
            marker_obstacles.scale.x = 0.2;
            marker_obstacles.scale.y = 0.2;
            marker_obstacles.color.a = 1.0;
            marker_obstacles.color.g = 1.0;
            marker_obstacles.lifetime = ros::Duration(0.0);
            marker_obstacles.pose.orientation.w = 1.0;
            marker_obstacles.points.reserve(centers.size());

            for(const auto& center : centers)
            {
                geometry_msgs::Point temp_point;
                temp_point.x = center.x * static_cast<double>(map.resolution) + map.origin.position.x;
                temp_point.y = center.y * static_cast<double>(map.resolution) + map.origin.position.y;

                marker_obstacles.points.push_back(std::move(temp_point));
            }

            marker_array.markers.push_back(std::move(marker_obstacles));
            marker_array.markers.push_back(std::move(marker));
            marker_array.markers.push_back(std::move(marker_lonely));
            edges_viz_pub.publish(marker_array);
        }
    }

    void SharedVoronoiGlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros)
    {
        if (!initialized_)
        {
            //Read parameters
            nh.getParam("occupancy_threshold", occupancy_threshold);
            nh.getParam("update_voronoi_rate", update_voronoi_rate);
            nh.getParam("print_timings", print_timings);
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
            nh.getParam("joystick_topic", joystick_topic);
            nh.getParam("visualize_edges", visualize_edges);
            nh.getParam("node_connection_threshold_pix", node_connection_threshold_pix);
            nh.getParam("collision_threshold", collision_threshold);
            nh.getParam("joy_max_lin", joy_max_lin);            
            nh.getParam("joy_max_ang", joy_max_ang);

            voronoi_path.h_class_threshold = h_class_threshold;
            voronoi_path.print_timings = print_timings;
            voronoi_path.node_connection_threshold_pix = node_connection_threshold_pix;
            voronoi_path.extra_point_distance = extra_point_distance;
            voronoi_path.min_node_sep_sq = min_node_sep_sq;

            //Subscribe and advertise related topics
            global_costmap_sub = nh.subscribe("/move_base/global_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::globalCostmapCB, this);
            global_update_sub = nh.subscribe("/move_base/global_costmap/costmap_updates", 1, &SharedVoronoiGlobalPlanner::globalCostmapUpdateCB, this);
            local_costmap_sub = nh.subscribe("/move_base/local_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::localCostmapCB, this);
            user_vel_sub = nh.subscribe(joystick_topic, 1, &SharedVoronoiGlobalPlanner::cmdVelCB, this);

            global_path_pub = nh.advertise<nav_msgs::Path>("plan", 1);
            all_paths_pub = nh.advertise<visualization_msgs::MarkerArray>("all_paths", 1);
            user_direction_pub = nh.advertise<visualization_msgs::Marker>("user_direction", 1);
            edges_viz_pub = nh.advertise<visualization_msgs::MarkerArray>("voronoi_edges", 1);

            //Create timer to update Voronoi diagram
            voronoi_update_timer = nh.createWallTimer(ros::WallDuration(1.0 / update_voronoi_rate), &SharedVoronoiGlobalPlanner::updateVoronoiCB, this);

            ROS_INFO("Shared Voronoi Global Planner initialized");
        }

        else
            ROS_INFO("Shared Voronoi Global Planner already initialized, not doing anything");
    }

    bool SharedVoronoiGlobalPlanner::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal, std::vector<geometry_msgs::PoseStamped> &plan)
    {
        //Get start and end points in terms of global costmap pixels
        voronoi_path::GraphNode start_point((start.pose.position.x - map.origin.position.x) / map.resolution,
                                            (start.pose.position.y - map.origin.position.y) / map.resolution);
        voronoi_path::GraphNode end_point((goal.pose.position.x - map.origin.position.x) / map.resolution,
                                          (goal.pose.position.y - map.origin.position.y) / map.resolution);

        // //Reset previous path if goal has changed
        // if (prev_goal.header.frame_id.empty())
        //     prev_goal = goal;

        // else
        // {
        //     //If goal has changed
        //     if (prev_goal.pose.position.x != goal.pose.position.x ||
        //         prev_goal.pose.position.y != goal.pose.position.y ||
        //         prev_goal.pose.position.z != goal.pose.position.z)
        //     {
        //         prev_goal = goal;
        //         // prev_path.clear();
        //     }
        // }

        //Get voronoi paths
        std::vector<std::vector<voronoi_path::GraphNode>> all_paths = voronoi_path.getPath(start_point, end_point, num_paths);

        if (all_paths.size() < num_paths)
            ROS_WARN("Could not find all requested paths. Requested: %d, found: %ld", num_paths, all_paths.size());

        //If paths are found
        if (!all_paths.empty())
        {
            std::vector<std::vector<geometry_msgs::PoseStamped>> all_paths_meters(all_paths.size());
            visualization_msgs::MarkerArray marker_array;

            //Convert node numbers to position on map for path
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = map.frame_id;
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
                    new_pose.pose.position.x = all_paths[i][j].x * map.resolution + map.origin.position.x;
                    new_pose.pose.position.y = all_paths[i][j].y * map.resolution + map.origin.position.y;
                    new_pose.pose.position.z = 0;

                    //TODO: Set orientation of intermediate poses
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

            // //Prune previous selected path
            // while (!prev_path.empty())
            // {
            //     double dist = sqrt(pow(prev_path[0].pose.position.x - start.pose.position.x, 2) + pow(prev_path[0].pose.position.y - start.pose.position.y, 2));
            //     if (dist < 0.3)
            //         prev_path.erase(prev_path.begin());

            //     else
            //         break;
            // }

            // //Add previously selected path into selections
            // if(!prev_path.empty())
            //     all_paths_meters.push_back(prev_path);

            //Select the path most similar to user commanded velocity path
            int preferred_path = 0;
            double dist = pow(start.pose.position.x - goal.pose.position.x, 2) + pow(start.pose.position.y - goal.pose.position.y, 2);
            if ((cmd_vel.linear.x != 0.0 || cmd_vel.angular.z != 0.0) && dist > pow(near_goal_threshold, 2))
                preferred_path = getMatchedPath(start, all_paths_meters);

            //Set selected plan
            if (!all_paths_meters[preferred_path].empty())
            {
                plan = all_paths_meters[preferred_path];
                // prev_path = plan;
            }

            //Publish selected plan for visualization
            nav_msgs::Path viz_path;
            viz_path.header.stamp = ros::Time::now();
            viz_path.header.frame_id = map.frame_id;
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
        marker.header.frame_id = map.frame_id;
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

        //Normalize the cmd angular velocity to get accurate joystick direction
        double normalized_ang = cmd_vel.angular.z / joy_max_ang;
        if(normalized_ang > 1)
            normalized_ang = 1;

        //Normalize the cmd linear velocity to get accurate joystick direction
        double normalized_lin = cmd_vel.linear.x / joy_max_lin;
        if(normalized_lin > 1)
            normalized_lin = 1;

        double new_local_dir = user_dir_filter * atan2(normalized_ang, normalized_lin) + (1 - user_dir_filter) * prev_local_dir;
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
            double rel_local_x = -map.origin.position.x + local_costmap.info.origin.position.x;
            double rel_local_y = -map.origin.position.y + local_costmap.info.origin.position.y;

            //Costmap is rotated ccw 90deg in rviz
            //Convert distance to pixels in global costmap resolution
            int x_pixel_offset = rel_local_x / map.resolution;
            int y_pixel_offset = rel_local_y / map.resolution;

            std::vector<voronoi_path::GraphNode> local_vertices;
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset, y_pixel_offset));
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset + local_costmap.info.width, y_pixel_offset));
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset + local_costmap.info.width, y_pixel_offset + local_costmap.info.height));
            local_vertices.push_back(voronoi_path::GraphNode(x_pixel_offset, y_pixel_offset + local_costmap.info.height));

            if (add_local_costmap_corners)
                voronoi_path.setLocalVertices(local_vertices);

            //Restore modified global costmap pixels to old value in previous loop, in cases when local obstacle is moving
            for (int i = 0; i < map_pixels_backup.size(); ++i)
                map.data[map_pixels_backup[i].first] = map_pixels_backup[i].second;

            map_pixels_backup.clear();

            //Copy data to internal map storage from local costmap if the pixel surpasses an occupancy threshold
            for (int i = 0; i < local_costmap.data.size(); ++i)
            {
                int local_data = local_costmap.data[i];

                if (local_data >= occupancy_threshold)
                {
                    int global_curr_x = i % local_costmap.info.width + x_pixel_offset;
                    int global_curr_y = i / local_costmap.info.width + y_pixel_offset;
                    map_pixels_backup.emplace_back(global_curr_y * map.width + global_curr_x,
                                                   map.data[global_curr_y * map.width + global_curr_x]);
                    map.data[global_curr_y * map.width + global_curr_x] = local_data;
                }
            }
        }
    }

    void SharedVoronoiGlobalPlanner::globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        //Copy all required data to internal map storage
        map.height = msg->info.height;
        map.width = msg->info.width;
        map.frame_id = msg->header.frame_id;
        map.resolution = msg->info.resolution;
        map.origin.position.x = msg->info.origin.position.x;
        map.origin.position.y = msg->info.origin.position.y;
        map.data = msg->data;
    }

    void SharedVoronoiGlobalPlanner::globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg)
    {
        //Assign update of map data to local map copy
        map.data = msg->data;

        //Call local costmap cb to make sure that local obstacles are not overwritten by global costmap update
        if(!map.data.empty())
        {
            const auto costmap_ptr = boost::make_shared<nav_msgs::OccupancyGrid>(local_costmap);
            localCostmapCB(costmap_ptr);
        }
    }

    void SharedVoronoiGlobalPlanner::cmdVelCB(const geometry_msgs::Twist::ConstPtr &msg)
    {
        cmd_vel = *msg;
    }
} // namespace shared_voronoi_global_planner