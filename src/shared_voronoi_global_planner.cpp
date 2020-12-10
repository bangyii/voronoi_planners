#include "shared_voronoi_global_planner.h"
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <limits>
#include <geometry_msgs/Point.h>
#include <tf2_ros/transform_listener.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <shared_voronoi_global_planner/AdjacencyList.h>
#include <shared_voronoi_global_planner/AdjacencyNodes.h>
#include <shared_voronoi_global_planner/NodeInfo.h>
#include <shared_voronoi_global_planner/NodeInfoList.h>
#include <shared_voronoi_global_planner/SortedNodeInfo.h>
#include <shared_voronoi_global_planner/SortedNodesList.h>

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
        if (update_voronoi_rate == 0)
        {
            ros::Rate r(1);
            while (map.data.empty())
            {
                ROS_WARN("Map is still empty, unable to initialize, waiting until map is not empty");
                r.sleep();
                ros::spinOnce();
            }

            ROS_WARN("Voronoi diagram initialized");
        }

        else
        {
            if (map.data.empty())
            {
                ROS_WARN("Map is still empty, skipping update of voronoi diagram");
                return;
            }
        }

        //Call voronoi object to update its internal voronoi diagram
        voronoi_path.mapToGraph(&map);
        nav_msgs::OccupancyGrid temp_map;
        temp_map.data = map.data;
        temp_map.info.resolution = map.resolution;
        temp_map.info.width = map.width;
        temp_map.info.height = map.height;
        temp_map.header.frame_id = "map";
        temp_map.header.stamp = ros::Time::now();
        temp_map.info.origin.position.x = -100;
        temp_map.info.origin.position.y = -100;
        temp_map.info.origin.orientation.w = 1.0;
        costmap_pub.publish(temp_map);

        //Publish adjacency list and corresponding info to 
        std::vector<std::vector<int>> adj_list_raw = voronoi_path.getAdjList();
        std::vector<voronoi_path::GraphNode> node_inf_raw = voronoi_path.getNodeInfo();
        shared_voronoi_global_planner::AdjacencyList adj_list;
        shared_voronoi_global_planner::NodeInfoList node_info;
        adj_list.nodes.resize(adj_list_raw.size());
        node_info.node_info.resize(node_inf_raw.size());
        for(int i = 0; i < node_inf_raw.size(); ++i)
        {
            adj_list.nodes[i].adjacent_nodes = adj_list_raw[i];
            node_info.node_info[i].x = node_inf_raw[i].x * static_cast<double>(map.resolution) + map.origin.position.x;
            node_info.node_info[i].y = node_inf_raw[i].y * static_cast<double>(map.resolution) + map.origin.position.y;
        }

        adjacency_list_pub.publish(adj_list);
        node_info_pub.publish(node_info);

        //Publish visualization marker for use in rviz
        if (visualize_edges)
        {
            std::vector<voronoi_path::GraphNode> nodes;
            std::vector<voronoi_path::GraphNode> lonely_nodes;
            std::vector<voronoi_path::GraphNode> centers;
            voronoi_path.getObstacleCentroids(centers);
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
            marker.pose.orientation.w = 1.0;
            marker.points.reserve(nodes.size());

            for (const auto &node : nodes)
            {
                geometry_msgs::Point temp_point;
                temp_point.x = node.x * static_cast<double>(map.resolution) + map.origin.position.x;
                temp_point.y = node.y * static_cast<double>(map.resolution) + map.origin.position.y;

                if (node.x > 0 && node.x < 0.01 && node.y > 0 && node.y < 0.01)
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
            marker_lonely.pose.orientation.w = 1.0;
            marker_lonely.points.reserve(lonely_nodes.size());

            for (const auto &node : lonely_nodes)
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
            marker_obstacles.pose.orientation.w = 1.0;
            marker_obstacles.points.reserve(centers.size());

            for (const auto &center : centers)
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
            nh.getParam("forward_sim_resolution", forward_sim_resolution);
            nh.getParam("num_paths", num_paths);
            nh.getParam("publish_all_path_markers", publish_all_path_markers);
            nh.getParam("joystick_topic", joystick_topic);
            nh.getParam("visualize_edges", visualize_edges);
            nh.getParam("node_connection_threshold_pix", node_connection_threshold_pix);
            nh.getParam("collision_threshold", collision_threshold);
            nh.getParam("joy_max_lin", joy_max_lin);
            nh.getParam("joy_max_ang", joy_max_ang);
            nh.getParam("subscribe_local_costmap", subscribe_local_costmap);
            nh.getParam("trimming_collision_threshold", trimming_collision_threshold);
            nh.getParam("search_radius", search_radius);
            nh.getParam("selection_threshold", selection_threshold);
            nh.getParam("static_global_map", static_global_map);
            nh.getParam("xy_goal_tolerance", xy_goal_tolerance);
            nh.getParam("odom_topic", odom_topic);
            nh.getParam("sorted_nodes_dist_thresh", sorted_nodes_dist_thresh);

            //Set parameters for voronoi path object
            voronoi_path.h_class_threshold = h_class_threshold;
            voronoi_path.print_timings = print_timings;
            voronoi_path.node_connection_threshold_pix = node_connection_threshold_pix;
            voronoi_path.extra_point_distance = extra_point_distance;
            voronoi_path.min_node_sep_sq = min_node_sep_sq;
            voronoi_path.trimming_collision_threshold = trimming_collision_threshold;
            voronoi_path.search_radius = search_radius;
            voronoi_path.open_cv_scale = open_cv_scale;

            //Subscribe and advertise related topics
            global_costmap_sub = nh.subscribe("/move_base/global_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::globalCostmapCB, this);

            if (!static_global_map)
                global_update_sub = nh.subscribe("/move_base/global_costmap/costmap_updates", 1, &SharedVoronoiGlobalPlanner::globalCostmapUpdateCB, this);

            if (subscribe_local_costmap)
                local_costmap_sub = nh.subscribe("/move_base/local_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::localCostmapCB, this);
                
            //Subscribe to joystick output to get direction selected by user
            user_vel_sub = nh.subscribe(joystick_topic, 1, &SharedVoronoiGlobalPlanner::cmdVelCB, this);

            //Subscribe to odometry to make sure that sorted node list is updated
            odom_sub = nh.subscribe(odom_topic, 1, &SharedVoronoiGlobalPlanner::odomCB, this);

            //Publisher for chosen path, all paths, user's indicated direction, and voronoi graph edges for visualization respectively
            global_path_pub = nh.advertise<nav_msgs::Path>("plan", 1);
            all_paths_pub = nh.advertise<visualization_msgs::MarkerArray>("all_paths", 1);
            user_direction_pub = nh.advertise<visualization_msgs::Marker>("user_direction", 1);
            edges_viz_pub = nh.advertise<visualization_msgs::MarkerArray>("voronoi_edges", 1, true);
            adjacency_list_pub = nh.advertise<shared_voronoi_global_planner::AdjacencyList>("adjacency_list", 1, true);
            node_info_pub = nh.advertise<shared_voronoi_global_planner::NodeInfoList>("node_info", 1, true);
            sorted_nodes_pub = nh.advertise<shared_voronoi_global_planner::SortedNodesList>("sorted_nodes", 1);
            costmap_pub = nh.advertise<nav_msgs::OccupancyGrid>("grid", 1);

            //Create timer to update Voronoi diagram, use one shot timer if update rate is 0
            if (update_voronoi_rate != 0)
                voronoi_update_timer = nh.createWallTimer(ros::WallDuration(1.0 / update_voronoi_rate), &SharedVoronoiGlobalPlanner::updateVoronoiCB, this);
            else
                voronoi_update_timer = nh.createWallTimer(ros::WallDuration(1), &SharedVoronoiGlobalPlanner::updateVoronoiCB, this, true);

            ROS_INFO("Shared Voronoi Global Planner initialized");
        }

        else
            ROS_INFO("Shared Voronoi Global Planner already initialized, not doing anything");
    }

    bool SharedVoronoiGlobalPlanner::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal, std::vector<geometry_msgs::PoseStamped> &plan)
    {
        static std::vector<std::vector<voronoi_path::GraphNode>> all_paths;
        static std::vector<std::vector<geometry_msgs::PoseStamped>> all_paths_meters;

        //Transform goal and start to map frame if they are not already in map frame
        geometry_msgs::PoseStamped start_ = start;
        geometry_msgs::PoseStamped goal_ = goal;

        if (goal_.header.frame_id != map.frame_id)
        {
            tf2_ros::Buffer tf_buffer;
            tf2_ros::TransformListener tf_listener(tf_buffer);
            ROS_WARN("Goal position is not in map frame, transforming goal to map frame before continuing");
            geometry_msgs::TransformStamped goal2MapTF;

            try
            {
                goal2MapTF = tf_buffer.lookupTransform(map.frame_id, goal.header.frame_id, ros::Time(0), ros::Duration(1.0));
            }
            catch (tf2::TransformException &Exception)
            {
                ROS_ERROR_STREAM(Exception.what());
            }

            geometry_msgs::Pose temp_goal;
            temp_goal = goal_.pose;
            tf2::doTransform<geometry_msgs::Pose>(temp_goal, temp_goal, goal2MapTF);
            goal_.pose = temp_goal;
        }

        //Get start and end points in terms of global costmap pixels
        voronoi_path::GraphNode end_point((goal_.pose.position.x - map.origin.position.x) / map.resolution,
                                          (goal_.pose.position.y - map.origin.position.y) / map.resolution);
        voronoi_path::GraphNode start_point((start_.pose.position.x - map.origin.position.x) / map.resolution,
                                            (start_.pose.position.y - map.origin.position.y) / map.resolution);

        //Send previous time steps' paths when too near to goal if there already paths found
        double dist = sqrt(pow(start_.pose.position.x - goal_.pose.position.x, 2) + pow(start_.pose.position.y - goal_.pose.position.y, 2));
        if(dist < xy_goal_tolerance && all_paths_meters.size() > preferred_path)
        {
            plan = all_paths_meters[preferred_path];
        }

        //move_base had a goal previously set, so paths should be trimmed based on previous one instead of replanning entirely
        else if (voronoi_path.hasPreviousPaths() && prev_goal == end_point)
        {
            all_paths = voronoi_path.replan(start_point, end_point, num_paths, preferred_path);
            if (!voronoi_path.bezierInterp(all_paths))
                ROS_DEBUG("Bezier interpolation failed, original path already collides with obstacle");
        }

        //move_base was not running, there are no previous paths. So planning should be done from scratch
        else
        {
            //Clear all previous paths and preferences before getting new path
            voronoi_path.clearPreviousPaths();
            preferred_path = 0;
            all_paths = voronoi_path.getPath(start_point, end_point, num_paths);

            //TODO: What is the purpose of this?
            //Smooth the path received from voronoi planner, return true when fail so the global planner can try replanning/update position
            if (!voronoi_path.bezierInterp(all_paths))
                ROS_DEBUG("Bezier interpolation failed, original path already collides with obstacle");
        }

        if (all_paths.size() < num_paths)
            ROS_WARN("Could not find all requested paths. Requested: %d, found: %ld", num_paths, all_paths.size());

        //If paths are found
        if (!all_paths.empty())
        {
            all_paths_meters.clear();
            all_paths_meters.resize(all_paths.size());
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
                    marker.color.g = 0;
                    marker.color.b = (255 / all_paths.size() * (all_paths.size() - i)) / 255.0;
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

                if (publish_all_path_markers)
                    marker_array.markers.push_back(marker);

                //Adjust orientation of start and end positions
                if (!all_paths_meters[i].empty())
                {
                    all_paths_meters[i][0].pose.orientation = start_.pose.orientation;
                    all_paths_meters[i].back().pose.orientation = goal_.pose.orientation;
                }
            }

            //Publish visualization for all available paths
            if (publish_all_path_markers)
                all_paths_pub.publish(marker_array);

            //Select the path most similar to user commanded velocity path
            double dist = pow(start_.pose.position.x - goal_.pose.position.x, 2) + pow(start_.pose.position.y - goal_.pose.position.y, 2);
            if (sqrt(pow(cmd_vel.linear.x, 2) + pow(cmd_vel.angular.z, 2)) > 0.8 * joy_max_lin && dist > pow(near_goal_threshold, 2))
                preferred_path = getMatchedPath(start_, all_paths_meters);

            //Set selected plan
            if (all_paths_meters.size() > preferred_path)
                plan = all_paths_meters[preferred_path];

            //Publish selected plan for visualization
            nav_msgs::Path viz_path;
            viz_path.header.stamp = ros::Time::now();
            viz_path.header.frame_id = map.frame_id;
            viz_path.poses = plan;
            global_path_pub.publish(viz_path);
            prev_goal = end_point;

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
        if (normalized_ang > 1)
            normalized_ang = 1;

        //Normalize the cmd linear velocity to get accurate joystick direction
        double normalized_lin = cmd_vel.linear.x / joy_max_lin;
        if (normalized_lin > 1)
            normalized_lin = 1;

        double theta = tf::getYaw(curr_pose.pose.orientation) + atan2(normalized_ang, normalized_lin);

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

        double max_s = sqrt(pow(user_path.back().first - user_path[0].first, 2) +
                            pow(user_path.back().second - user_path[0].second, 2));

        std::vector<double> ang_diff_sq(plans_.size(), 0);

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

        //If the angular difference of a path is greater than selection_threshold%, change cost to infinity
        std::vector<double> total_costs = voronoi_path.getAllPathCosts();
        double min_val = *std::min_element(ang_diff_sq.begin(), ang_diff_sq.end());
        for (int i = 0; i < ang_diff_sq.size(); ++i)
        {
            if (ang_diff_sq[i] / min_val >= selection_threshold)
                total_costs[i] = std::numeric_limits<double>::infinity();
        }

        return std::min_element(total_costs.begin(), total_costs.end()) - total_costs.begin();
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
        //Assign update of map data to local copy of map
        map.data = msg->data;

        //Call local costmap cb to make sure that local obstacles are not overwritten by global costmap update
        if (!map.data.empty())
        {
            const auto costmap_ptr = boost::make_shared<nav_msgs::OccupancyGrid>(local_costmap);
            localCostmapCB(costmap_ptr);
        }
    }

    void SharedVoronoiGlobalPlanner::cmdVelCB(const geometry_msgs::Twist::ConstPtr &msg)
    {
        cmd_vel = *msg;
    }

    void SharedVoronoiGlobalPlanner::odomCB(const nav_msgs::Odometry::ConstPtr &msg)
    {
        //last_sorted_position pose will be 0 if it was not initialized before
        double dist = pow(msg->pose.pose.position.x - last_sorted_position.pose.pose.position.x, 2) +
                            pow(msg->pose.pose.position.y - last_sorted_position.pose.pose.position.y, 2);

        //Greater than threshold, time to update sorted nodes list
        if(last_sorted_position.header.frame_id.empty() || dist > pow(sorted_nodes_dist_thresh, 2))
        {
            sorted_nodes_raw = voronoi_path.getSortedNodeList(voronoi_path::GraphNode(msg->pose.pose.position.x, msg->pose.pose.position.y));

            if(!sorted_nodes_raw.empty())
                last_sorted_position = *msg;

            //Publish sorted vector of nodes that are nearby, distance is in square meters
            shared_voronoi_global_planner::SortedNodesList sorted_nodes;
            sorted_nodes.sorted_nodes.resize(sorted_nodes_raw.size());
            for(int i = 0; i <  sorted_nodes_raw.size(); ++i)
            {
                sorted_nodes.sorted_nodes[i].node = sorted_nodes_raw[i].second;
                sorted_nodes.sorted_nodes[i].distance = sorted_nodes_raw[i].first * map.resolution * map.resolution;
            }
            sorted_nodes_pub.publish(sorted_nodes);
        }
    }
} // namespace shared_voronoi_global_planner
