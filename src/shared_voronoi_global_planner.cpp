#include "shared_voronoi_global_planner.h"
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>
#include <thread>
#include <future>

PLUGINLIB_EXPORT_CLASS(shared_voronoi_global_planner::SharedVoronoiGlobalPlanner, nav_core::BaseGlobalPlanner)

namespace shared_voronoi_global_planner
{
    SharedVoronoiGlobalPlanner::SharedVoronoiGlobalPlanner() : nh("~")
    {
    }

    SharedVoronoiGlobalPlanner::SharedVoronoiGlobalPlanner(std::string name, costmap_2d::Costmap2DROS *costmap_ros) : nh("~")
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
        merged_costmap_pub.publish(merged_costmap);
    }

    void SharedVoronoiGlobalPlanner::updateVoronoiMapCB(const ros::WallTimerEvent &e)
    {
        map.height = merged_costmap.info.height;
        map.width = merged_costmap.info.width;
        map.frame_id = merged_costmap.header.frame_id;
        map.resolution = merged_costmap.info.resolution;

        map.data.clear();
        map.data.insert(map.data.begin(), merged_costmap.data.begin(), merged_costmap.data.end());
        threadedMapCleanup();
    }

    void SharedVoronoiGlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros)
    {
        if (!initialized_)
        {
            //Subscribe and advertise related topics
            local_costmap_sub = nh.subscribe("/move_base/local_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::localCostmapCB, this);
            global_costmap_sub = nh.subscribe("/move_base/global_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::globalCostmapCB, this);
            global_update_sub = nh.subscribe("/move_base/global_costmap/costmap_updates", 1, &SharedVoronoiGlobalPlanner::globalCostmapUpdateCB, this);

            merged_costmap_pub = nh.advertise<nav_msgs::OccupancyGrid>("merged_costmap", 1);
            global_path_pub = nh.advertise<nav_msgs::Path>("plan", 1);
            alternate_path_pub = nh.advertise<nav_msgs::Path>("alternate_plan", 1);

            //Create timer to update Voronoi diagram
            voronoi_update_timer = nh.createWallTimer(ros::WallDuration(1.0 / update_voronoi_rate), &SharedVoronoiGlobalPlanner::updateVoronoiCB, this);
            map_update_timer = nh.createWallTimer(ros::WallDuration(1.0/update_costmap_rate), &SharedVoronoiGlobalPlanner::updateVoronoiMapCB, this);
            // prev_costmap_time = std::chrono::system_clock::now();

            ROS_INFO("Shared Voronoi Global Planner initialized");
        }

        else
            ROS_INFO("Shared Voronoi Global Planner already initialized, not doing anything");
    }

    bool SharedVoronoiGlobalPlanner::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal, std::vector<geometry_msgs::PoseStamped> &plan)
    {
        // voronoi_path.findObstacleCentroids();
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

            ROS_INFO("Voronoi diagram is updating, waiting for it to complete");
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

            plan[0] = start;
            plan.back() = goal;

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

                alt_plan[0] = start;
                alt_plan.back() = goal;
            }

            //TODO: Check for collision between start and first, and end and last node

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

    void SharedVoronoiGlobalPlanner::localCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        local_costmap = *msg;

        //Merge costmaps if global map is not empty
        //TODO: Can be threaded to increase speed
        if (!map.data.empty())
        {
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
                voronoi_path.setLocalVertices(local_vertices);

                for (int i = 0; i < local_costmap.data.size(); i++)
                {
                    int local_data = local_costmap.data[i];

                    if (local_data > costmap_threshold)
                    {
                        int global_curr_x = i % local_costmap.info.width + x_pixel_offset;
                        int global_curr_y = i / local_costmap.info.width + y_pixel_offset;
                        map.data[global_curr_y * merged_costmap.info.width + global_curr_x] = local_data;
                    }
                }
            }

            // merged_costmap.data.clear();
            // merged_costmap.data.insert(merged_costmap.data.begin(), map.data.begin(), map.data.end());

            // merged_costmap_pub.publish(merged_costmap);
        }
    }

    void SharedVoronoiGlobalPlanner::globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        // if ((std::chrono::system_clock::now() - prev_costmap_time).count() / 1000000000.0 < (1.0 / update_costmap_rate))
        //     return;

        merged_costmap = *msg;

        // //Initialize voronoi graph
        // map.height = merged_costmap.info.height;
        // map.width = merged_costmap.info.width;
        // map.frame_id = merged_costmap.header.frame_id;
        // map.resolution = merged_costmap.info.resolution;

        // map.data.clear();
        // map.data.insert(map.data.begin(), merged_costmap.data.begin(), merged_costmap.data.end());
        // threadedMapCleanup();

        // prev_costmap_time = std::chrono::system_clock::now();
    }

    void SharedVoronoiGlobalPlanner::globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg)
    {
        // if ((std::chrono::system_clock::now() - prev_costmap_time).count() / 1000000000.0 < (1.0 / update_costmap_rate))
        //     return;

        // map.data.clear();
        // map.data.insert(map.data.begin(), msg->data.begin(), msg->data.end());
        // threadedMapCleanup();

        // prev_costmap_time = std::chrono::system_clock::now();
        merged_costmap.data = msg->data;
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

        // std::cout << "Time taken to cleanup map " << (std::chrono::system_clock::now() - map_cleanup_time).count() / 1000000000.0 << std::endl;
    }
} // namespace shared_voronoi_global_planner