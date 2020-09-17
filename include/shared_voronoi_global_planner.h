#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include "voronoi_path.h"

#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <angles/angles.h>
#include <base_local_planner/world_model.h>
#include <base_local_planner/costmap_model.h>
#include <chrono>

#ifndef SHARED_VORONOI_GLOBAL_PLANNER_H
#define SHARED_VORONOI_GLOBAL_PLANNER_H

namespace shared_voronoi_global_planner
{
    class SharedVoronoiGlobalPlanner : public nav_core::BaseGlobalPlanner
    {
    public:
        SharedVoronoiGlobalPlanner();
        SharedVoronoiGlobalPlanner(std::string name, costmap_2d::Costmap2DROS *costmap_ros);
        /** overridden classes from interface nav_core::BaseGlobalPlanner **/
        void initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros);
        bool makePlan(const geometry_msgs::PoseStamped &start,
                      const geometry_msgs::PoseStamped &goal,
                      std::vector<geometry_msgs::PoseStamped> &plan);

    private:
        nav_msgs::OccupancyGrid local_costmap;
        nav_msgs::OccupancyGrid merged_costmap;
        int costmap_threshold = 90;
        bool initialized_ = false;
        voronoi_path::Map map;
        voronoi_path::voronoi_path voronoi_path;
        bool map_received = false;
        double update_voronoi_rate = 0.3;

        std::chrono::time_point<std::chrono::system_clock> prev_set_map_time;

        ros::NodeHandle nh;
        ros::Subscriber local_costmap_sub;
        ros::Subscriber global_costmap_sub;
        ros::Subscriber global_update_sub;
        ros::Publisher merged_costmap_pub;
        ros::Publisher global_path_pub;
        ros::Publisher alternate_path_pub;

        void localCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg);
        void globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg);
        void globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg);
    };
}; // namespace shared_voronoi_global_planner

#endif