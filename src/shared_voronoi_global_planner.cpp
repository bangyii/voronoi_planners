#include "shared_voronoi_global_planner.h"
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>
#include <thread>
#include <future>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <limits>

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

        voronoi_path.print_timings = print_timings;
        voronoi_path.mapToGraph(map);
    }

    void SharedVoronoiGlobalPlanner::updateVoronoiMapCB(const ros::WallTimerEvent &e)
    {
        map.height = merged_costmap.info.height;
        map.width = merged_costmap.info.width;
        map.frame_id = merged_costmap.header.frame_id;
        map.resolution = merged_costmap.info.resolution;
        map.origin.position.x = merged_costmap.info.origin.position.x;
        map.origin.position.y = merged_costmap.info.origin.position.y;

        // unsigned char *costmap_char = costmap->getCostmap()->getCharMap();
        // int pixels = costmap->getCostmap()->getSizeInCellsX() * costmap->getCostmap()->getSizeInCellsY();
        // merged_costmap.header.frame_id = costmap->getGlobalFrameID();
        // merged_costmap.header.stamp = ros::Time::now();
        // merged_costmap.info.origin.position.x = costmap->getCostmap()->getOriginX();
        // merged_costmap.info.origin.position.y = costmap->getCostmap()->getOriginY();
        // merged_costmap.info.origin.orientation.w = 1.0;
        // merged_costmap.info.resolution = costmap->getCostmap()->getResolution();
        // merged_costmap.info.width = costmap->getCostmap()->getSizeInCellsX();
        // merged_costmap.info.height = costmap->getCostmap()->getSizeInCellsY();

        // merged_costmap.data.clear();
        // for (int i = 0; i < pixels; ++i)
        // {
        //     // if(costmap_char[i] != -1 && costmap_char[i] != 255)
        //     // std::cout << static_cast<int>(costmap_char[i]) << "\n";

        //     if (static_cast<int>(costmap_char[i]) < 0)
        //         std::cout << "Negative map value!\n";

        //     merged_costmap.data.push_back(static_cast<int>(costmap_char[i]));
        // }
        // // std::cout << "Merged costmap size " << merged_costmap.data.size() << "\n";
        // merged_costmap_pub.publish(merged_costmap);

        map.data.clear();
        map.data.insert(map.data.begin(), merged_costmap.data.begin(), merged_costmap.data.end());
        threadedMapCleanup();
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
            nh.getParam("hash_resolution", hash_resolution);
            nh.getParam("hash_length", hash_length);
            nh.getParam("line_check_resolution", line_check_resolution);
            nh.getParam("pixels_to_skip", pixels_to_skip);
            nh.getParam("open_cv_scale", open_cv_scale);
            nh.getParam("h_class_threshold", h_class_threshold);
            nh.getParam("min_node_sep_sq", min_node_sep_sq);
            nh.getParam("extra_point_distance", extra_point_distance);
            nh.getParam("add_local_costmap_corners", add_local_costmap_corners);

            //Subscribe and advertise related topics
            global_costmap_sub = nh.subscribe("/move_base/global_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::globalCostmapCB, this);
            global_update_sub = nh.subscribe("/move_base/global_costmap/costmap_updates", 1, &SharedVoronoiGlobalPlanner::globalCostmapUpdateCB, this);
            local_costmap_sub = nh.subscribe("/move_base/local_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::localCostmapCB, this);
            user_vel_sub = nh.subscribe("/test_vel", 1, &SharedVoronoiGlobalPlanner::cmdVelCB, this);

            merged_costmap_pub = nh.advertise<nav_msgs::OccupancyGrid>("merged_costmap", 1);
            global_path_pub = nh.advertise<nav_msgs::Path>("plan", 1);
            alternate_path_pub = nh.advertise<nav_msgs::Path>("alternate_plan", 1);
            // centroid_pub = nh.advertise<nav_msgs::Path>("/centroids", 1);

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
        // //Centers are in terms of pixel of original image. Not in meters
        // std::vector<voronoi_path::GraphNode> centers = voronoi_path.findObstacleCentroids();
        // nav_msgs::Path centroid_path;
        // centroid_path.header.stamp = ros::Time::now();
        // centroid_path.header.frame_id = start.header.frame_id;
        // for(auto centroids : centers)
        // {
        //     geometry_msgs::PoseStamped new_pose;
        //     new_pose.header = centroid_path.header;
        //     new_pose.pose.orientation.w = 1;

        //     new_pose.pose.position.x = centroids.x * map.resolution - map.origin.position.x;
        //     new_pose.pose.position.y = centroids.y * map.resolution - map.origin.position.y;
        //     new_pose.pose.position.z = 0;

        //     if(isnan(new_pose.pose.position.x))
        //         new_pose.pose.position.x = -100;

        //     if(isnan(new_pose.pose.position.y))
        //         new_pose.pose.position.y = -100;

        //     centroid_path.poses.push_back(new_pose);
        // }
        // std::cout << "Publishing centroid path\n";
        // centroid_pub.publish(centroid_path);

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

        //Get voronoi paths
        std::vector<std::vector<voronoi_path::GraphNode>> all_paths = voronoi_path.getPath(start_point, end_point, num_paths);
        std::vector<geometry_msgs::PoseStamped> alt_plan;

        //If paths are found
        if (!all_paths.empty())
        {
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = merged_costmap.header.frame_id;

            for (int i = 0; i < all_paths[0].size(); i++)
            {
                geometry_msgs::PoseStamped new_pose;
                new_pose.header = header;
                new_pose.pose.position.x = all_paths[0][i].x * merged_costmap.info.resolution + merged_costmap.info.origin.position.x;
                new_pose.pose.position.y = all_paths[0][i].y * merged_costmap.info.resolution + merged_costmap.info.origin.position.y;
                new_pose.pose.position.z = 0;

                new_pose.pose.orientation.w = 1;

                plan.push_back(new_pose);
            }

            if (!plan.empty())
            {
                plan[0] = start;
                plan.back() = goal;
            }

            if (all_paths.size() == 2)
            {
                for (int i = 0; i < all_paths[1].size(); i++)
                {
                    geometry_msgs::PoseStamped new_pose;
                    new_pose.header = header;
                    new_pose.pose.position.x = all_paths[1][i].x * merged_costmap.info.resolution + merged_costmap.info.origin.position.x;
                    new_pose.pose.position.y = all_paths[1][i].y * merged_costmap.info.resolution + merged_costmap.info.origin.position.y;
                    new_pose.pose.position.z = 0;

                    new_pose.pose.orientation.w = 1;

                    alt_plan.push_back(new_pose);
                }

                if (!alt_plan.empty())
                {
                    alt_plan[0] = start;
                    alt_plan.back() = goal;
                }
            }

            //TODO: Check for collision between start and first, and end and last node

            //Select the path most similar to user commanded velocity path
            int preferred_path = 0;
            if (cmd_vel.linear.x != 0)
            {
                std::cout << "User input received, selecting most similar path\n";
                std::vector<std::vector<geometry_msgs::PoseStamped>> all_plans = {plan, alt_plan};
                preferred_path = getMatchedPath(start, all_plans);
                std::cout << preferred_path << std::endl;
            }

            if (preferred_path != 0)
            {
                std::swap(plan, alt_plan);
            }

            //Publish plan for visualization
            nav_msgs::Path viz_path;
            viz_path.header.stamp = ros::Time::now();
            viz_path.header.frame_id = merged_costmap.header.frame_id;
            viz_path.poses = plan;
            global_path_pub.publish(viz_path);

            viz_path.poses = alt_plan;
            alternate_path_pub.publish(viz_path);

            return true;
        }

        else
            return false;
    }

    int SharedVoronoiGlobalPlanner::getMatchedPath(const geometry_msgs::PoseStamped &curr_pose,
                                                   const std::vector<std::vector<geometry_msgs::PoseStamped>> &plans_)
    {
        //Forward simulate commanded velocity
        double lin_x = 0.2;
        double time_interval = fabs(forward_sim_resolution / lin_x);

        std::vector<std::pair<double, double>> user_path;
        user_path.reserve(int(forward_sim_time / time_interval));

        double x = curr_pose.pose.position.x;
        double y = curr_pose.pose.position.y;
        double theta = tf::getYaw(curr_pose.pose.orientation) + atan2(cmd_vel.angular.z, cmd_vel.linear.x);
        user_path.emplace_back(x, y);

        double curr_time = 0.0;
        while (curr_time <= forward_sim_time + 0.01)
        {
            if (curr_time > forward_sim_time)
                curr_time = forward_sim_time;

            x += lin_x * cos(theta) * time_interval;
            y += lin_x * sin(theta) * time_interval;
            // theta += cmd_vel.angular.z * time_interval;

            user_path.emplace_back(x, y);

            curr_time += time_interval;
        }

        double user_areas = 0;
        double y_offset = merged_costmap.info.origin.position.y;
        double total_dx = fabs(user_path.back().first - user_path[0].first);

        //Numerical integration for area under forward simulated path
        for (int i = 0; i < user_path.size() - 1; ++i)
        {
            //Calculate area under user path
            double user_dx = user_path[i + 1].first - user_path[i].first;

            //Shift all coordinates into positive y quadrant, using the origin of global map as offset
            double user_average_y = (user_path[i + 1].second - y_offset + user_path[i].second - y_offset) / 2.0;
            user_areas += user_dx * user_average_y;
        }

        //Numerical integration for area of possible plans
        std::vector<double> path_areas(plans_.size(), 0);
        for (int i = 0; i < plans_.size(); i++)
        {
            double dx_sum = 0;
            int j = 0;
            //TODO: dx sum tolerance should be a variable
            while (dx_sum <= total_dx + 0.1 && j + 1 < plans_[i].size())
            {
                double temp_dx = plans_[i][j + 1].pose.position.x - plans_[i][j].pose.position.x;
                double average_y = (plans_[i][j + 1].pose.position.y - y_offset + plans_[i][j + 1].pose.position.y - y_offset) / 2.0;
                path_areas[i] += temp_dx * average_y;

                dx_sum += fabs(temp_dx);
                ++j;
            }
        }

        //Return index of path that has the most similar area to the forward simulated path
        if (!path_areas.empty())
        {
            auto min_ind = std::min_element(path_areas.begin(), path_areas.end());
            return min_ind - path_areas.begin();
        }

        else
            return -1;
    }

    void SharedVoronoiGlobalPlanner::localCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
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
        // local_costmap_updated = true;
        merged_costmap_pub.publish(merged_costmap);
    }

    void SharedVoronoiGlobalPlanner::globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        merged_costmap = *msg;
    }

    void SharedVoronoiGlobalPlanner::globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg)
    {
        merged_costmap.data = msg->data;
            
        //Copy back local costmap pixels to prevent oscillation of existence in local obstacles
        if(!merged_costmap.data.empty())
        {
            if(!local_costmap.data.empty())
            {
                const auto costmap_ptr = boost::make_shared<nav_msgs::OccupancyGrid>(local_costmap);
                localCostmapCB(costmap_ptr);
            }
        }
    }

    void SharedVoronoiGlobalPlanner::cmdVelCB(const geometry_msgs::Twist::ConstPtr &msg)
    {
        cmd_vel = *msg;
    }

    void SharedVoronoiGlobalPlanner::threadedMapCleanup()
    {
        auto map_cleanup_time = std::chrono::system_clock::now();
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<std::vector<int>>> future_vector;
        future_vector.reserve(num_threads);

        int size = map.data.size();
        int num_pixels = floor(size / num_threads);
        int start_pixel = 0;

        for (int i = 0; i < num_threads; i++)
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
            start_pixel = i * num_pixels;
            future_vector[i].wait();
            std::vector<int> temp_vec = future_vector[i].get();

            map.data.insert(map.data.end(), make_move_iterator(temp_vec.begin()), make_move_iterator(temp_vec.end()));
        }
    }
} // namespace shared_voronoi_global_planner