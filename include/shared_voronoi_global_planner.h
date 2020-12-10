#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include "voronoi_path.h"

#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <angles/angles.h>
#include <base_local_planner/world_model.h>
#include <base_local_planner/costmap_model.h>
#include <chrono>
#include <memory>
#include <mutex>

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
        /**
         * Internal copy of local costmap from ROS
         **/
        nav_msgs::OccupancyGrid local_costmap;

        /** 
         * Internal map which merges global and local costmap from ros
         **/
        voronoi_path::Map map;

        /**
         * Voronoi path object which is used for planning
         **/
        voronoi_path::voronoi_path voronoi_path;

        /**
         * Flag indicating whether the planner has been initialized
         **/ 
        bool initialized_ = false;

        /**
         * Param to indicate number of paths to find
         **/
        int num_paths = 2;

        /**
         * Rate at which to update the voronoi diagram, Hz
         **/
        double update_voronoi_rate = 0.3;

        /**
         * Flag indicating whether to print timings, used for debugging
         **/
        bool print_timings = true;

        /**
         * If there is a node within this threshold away from a node that only has 1 connection, they will both be connected
         **/
        int node_connection_threshold_pix = 1;

        /**
         * Pixel resolution to increment when checking if an edge collision occurs. Value of 0.1 means the edge will
         * be checked at every 0.1 pixel intervals
         **/
        double line_check_resolution = 0.1;

        /**
         * Threhsold before a pixel is considered occupied. If pixel value is < occupancy_threshold, it is considered free
         **/
        int occupancy_threshold = 100;

        /**
         * Threshold before a pixel is considered occupied during collision checking, this is same as occupancy_threshold but 
         * collision_threshold is used when checking if an edge collides with obstacles. Can be used in conjunction with
         * ROS's costmap inflation to prevent planner from planning in between narrow spaces
         **/
        int collision_threshold = 85;

        /**
         * Radius to search around robot location to try and find an empty cell to connect to start of previous path, meters
         **/
        double search_radius = 1.0;

        /**
         * Threshold used for trimming paths, should be smaller than collision_threshold to prevent robot from getting stuck
         **/
        int trimming_collision_threshold = 75;

        /**
         * Pixels to skip during the reading of map to generate voronoi graph. Increasing pixels to skip reduces computation time
         * of voronoi generation, but also reduces voronoi diagram density, likely causing path finding issues
         **/
        int pixels_to_skip = 0;

        /**
         * Downscale factor used for scaling map before finding contours. Smaller values increase speed (possibly marginal)
         * but may decrease the accuracy of the centroids found
         **/
        double open_cv_scale = 0.25;

        /**
         * Threshold to classify a homotopy class as same or different. Ideally, same homotopy classes should have identical 
         * compelx values, but since "double" representation is used, some difference might be present for same homotopy classes
         **/
        double h_class_threshold = 0.2;

        /**
         * Minimum separation between nodes. If nodes are less than this value (m^2) apart, they will be cleaned up
         **/
        double min_node_sep_sq = 1.0;

        /**
         * Distance (m) to put the extra point which is used to ensure continuity
         **/
        double extra_point_distance = 1.0;

        /**
         * Joystick maximum linear velocity, to normalize for joystick direction
         **/
        double joy_max_lin = 1.0;

        /**
         * Joystick maximum angular velocity, to normalize for joystick direction
         **/
        double joy_max_ang = 1.0;

        /**
         * Time to forward simulate user's joystick input to select path
         **/
        double forward_sim_time = 1.0;       //s

        /**
         * Resolution in which to forward simulate the user's chosen direction
         **/ 
        double forward_sim_resolution = 0.1; //m

        /**
         * If the user is within this threshold (meters) from the goal, the user can no longer select a different path
         **/
        double near_goal_threshold = 1.0;

        /**
         * When the robot is within this threshold distance from the goal, no more replanning wil be done if there are already paths.
         * Previous paths will be returned instead. This is to allow move_base to trigger "goal reached" by reducing resources
         * used by global planner
         **/
        double xy_goal_tolerance = 0.15;

        /**
         * If there are multiple paths that are less than this threshold (%) greater than the best matching path, then those
         * paths will also be considered during path selection. ie, if there are 4 paths, and user selects a direction, if the 
         * match cost of those 4 paths to the user's direction are [1, 1.1, 1.5, 2.3], then path 1 and 2 will be considered.
         * Within all the paths that are <1.2*minCost, the physically shortest path will be chose
         **/
        double selection_threshold = 1.2;

        /**
         * Choice to add the 4 corners of the local costmap as virtual obstacles, to enable generation of a voronoi diagram 
         * in an empty space
         **/
        bool add_local_costmap_corners = false;

        /**
         * Parameter to indicate wheteher to publish markers for all generated paths
         **/
        bool publish_all_path_markers = false;

        /**
         * Joystick topic to subscribe to for user's indicated direction
         **/
        std::string joystick_topic = "/joy_vel";

        /**
         * Odometry topic
         **/
        std::string odom_topic = "/odom";

        /**
         * Update sorted nodes list distance threshold. When robot is this distance away from the previous update position, then list will update again
         **/
        double sorted_nodes_dist_thresh = 0.3;

        /**
         * Whether to publish markers to visualize the voronoi diagram's edges, lonely nodes in the voronoi diagram (singly connected),
         * and centroids of obstacles
         **/
        bool visualize_edges = false;

        /**
         * Parameter to set whether or not to subscribe to local costmap
         **/
        bool subscribe_local_costmap = true;

        /**
         * Parameter to set whether global map is static (no mapping is being run)
         **/
        bool static_global_map = true;

        /**
         * Interval variable to store the preffered path from previous time step. Defaults to 0 which is the shortest path
         **/
        int preferred_path = 0;

        /**
         * Internal variable to store previous goal. This will help determine if the goal has changed and a full planning needs to be 
         * done, instead of replanning from previous time step
         **/
        voronoi_path::GraphNode prev_goal;

        /**
         * Last position where sorted node list was computed
         **/
        nav_msgs::Odometry last_sorted_position;

        /**
         * Vector storing the sorted nodes list
         **/
        std::vector<std::pair<double, int>> sorted_nodes_raw;
        
        /**
         * Internal store of pixels that were modified during the last local costmap callback, used to restore the pixels in the following 
         * time step when a new callback is called
         **/
        std::vector<std::pair<int, int>> map_pixels_backup;

        //ROS variables
        ros::NodeHandle nh;
        ros::Subscriber local_costmap_sub;
        ros::Subscriber global_costmap_sub;
        ros::Subscriber global_update_sub;
        ros::Subscriber user_vel_sub;
        ros::Subscriber odom_sub;
        ros::Subscriber move_base_stat_sub;

        ros::Publisher global_path_pub;
        ros::Publisher all_paths_pub;
        ros::Publisher user_direction_pub;
        ros::Publisher edges_viz_pub;
        ros::Publisher adjacency_list_pub;
        ros::Publisher sorted_nodes_pub;
        ros::Publisher node_info_pub;
        ros::Publisher costmap_pub;

        /**
         * Timer for updating the voronoi diagram, if specified in rosparams
         **/
        ros::WallTimer voronoi_update_timer;

        /**
         * To store user's indicated direction
         **/
        geometry_msgs::Twist cmd_vel;

        /**
         * Callback for local costmap, if subscribed
         **/
        void localCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg);

        /**
         * Callback for global costmap
         **/
        void globalCostmapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg);

        /**
         * Callback for global costmap updates, required if the global costmap is not static and is being updated
         **/
        void globalCostmapUpdateCB(const map_msgs::OccupancyGridUpdate::ConstPtr &msg);

        /**
         * Callback for Twist from joystick which indicates user direction
         **/
        void cmdVelCB(const geometry_msgs::Twist::ConstPtr &msg);

        /**
         * Callback for odometry
         **/
        void odomCB(const nav_msgs::Odometry::ConstPtr &msg);

        /**
         * Callback for timer event to periodically update the voronoi diagram, if update_voronoi_rate is > 0
         **/
        void updateVoronoiCB(const ros::WallTimerEvent &e);

        /**
         * Get the closes matching path to the users' current joystick direction
         * @param curr_pose the current pose of the robot
         * @param plans_ all the plans to try to match the user's direction to
         * @return integer index of the path that is the closes match in plans_ vector
         **/
        int getMatchedPath(const geometry_msgs::PoseStamped &curr_pose, const std::vector<std::vector<geometry_msgs::PoseStamped>> &plans_);

        /**
         * Returns the minimum angle between two vectors
         * @param vec1 an array of 2 doubles, x and y, of the first vector
         * @param vec2 an array of 2 doubles, x and y, of the second vector
         * @return minimum angle between the 2 vectors in rads
         **/
        double vectorAngle(const double vec1[2], const double vec2[2]);
    };
}; // namespace shared_voronoi_global_planner

#endif