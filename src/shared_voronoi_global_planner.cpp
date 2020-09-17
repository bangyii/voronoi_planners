#include "shared_voronoi_global_planner.h"
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>

PLUGINLIB_EXPORT_CLASS(shared_voronoi_global_planner::SharedVoronoiGlobalPlanner, nav_core::BaseGlobalPlanner)

namespace shared_voronoi_global_planner
{
    SharedVoronoiGlobalPlanner::SharedVoronoiGlobalPlanner() : nh("~")
    {
    }

    SharedVoronoiGlobalPlanner::SharedVoronoiGlobalPlanner(std::string name, costmap_2d::Costmap2DROS *costmap_ros) : nh("~")
    {
    }

    void SharedVoronoiGlobalPlanner::updateVoronoiCB(const ros::TimerEvent &e)
    {
        voronoi_path.print_timings = print_timings;
        voronoi_path.mapToGraph(map);
        merged_costmap_pub.publish(merged_costmap);
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
            voronoi_update_timer = nh.createTimer(ros::Duration(1.0 / update_voronoi_rate), &SharedVoronoiGlobalPlanner::updateVoronoiCB, this);

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
        // if (((std::chrono::system_clock::now() - prev_set_map_time).count() / 1000000000.0) < (1.0 / update_voronoi_rate))
        //     return;

        local_costmap = *msg;

        //Merge costmaps if global map is not empty
        if (!merged_costmap.data.empty())
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

                for (int i = 0; i < merged_costmap.data.size(); i++)
                {
                    int curr_x_pixel = i % merged_costmap.info.width;
                    int curr_y_pixel = i / merged_costmap.info.width;

                    int local_x_pixel = curr_x_pixel - x_pixel_offset;
                    int local_y_pixel = curr_y_pixel - y_pixel_offset;
                    int local_index = local_x_pixel + local_y_pixel * local_costmap.info.width;

                    if (local_index < local_costmap.data.size() && local_x_pixel < local_costmap.info.width && local_y_pixel < local_costmap.info.height)
                    {
                        int current_data = merged_costmap.data[i];
                        int local_data = local_costmap.data[local_index];

                        if (local_data > costmap_threshold)
                            merged_costmap.data[i] = local_data;
                    }
                }
            }
        }

        map.height = merged_costmap.info.height;
        map.width = merged_costmap.info.width;

        map.data.clear();
        map.data.insert(map.data.begin(), merged_costmap.data.begin(), merged_costmap.data.end());

        map.frame_id = merged_costmap.header.frame_id;
        map.resolution = merged_costmap.info.resolution;
    }

    void SharedVoronoiGlobalPlanner::globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        merged_costmap = *msg;
        merged_costmap_pub.publish(merged_costmap);

        //Initialize voronoi graph
        map.height = merged_costmap.info.height;
        map.width = merged_costmap.info.width;

        map.data.clear();
        map.data.insert(map.data.begin(), merged_costmap.data.begin(), merged_costmap.data.end());

        map.frame_id = merged_costmap.header.frame_id;
        map.resolution = merged_costmap.info.resolution;
    }

    void SharedVoronoiGlobalPlanner::globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg)
    {
        merged_costmap.data = msg->data;
    }
} // namespace shared_voronoi_global_planner