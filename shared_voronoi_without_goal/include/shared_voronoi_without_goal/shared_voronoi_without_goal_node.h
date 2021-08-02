#include <ros/ros.h>
#include <shared_voronoi_without_goal/voronoi_path.h>
#include <tf2_ros/transform_listener.h>
#include <string>
#include <vector>

Map map;
voronoi_path::VoronoiPath v_path;
ros::Subscriber global_map_sub;
ros::Subscriber move_base_cancel_sub;
ros::Publisher all_paths_pub;
ros::Publisher edges_viz_pub;
ros::Publisher map_pub;
ros::Publisher all_paths_ind_pub;
ros::WallTimer voronoi_update_timer;

/**
 * TF buffer and listener in global scope to reduce waiting time before a recent tf is received
 **/
tf2_ros::Buffer tf_buffer;

std::vector<voronoi_path::Path> all_paths;

double occupancy_threshold = 99;
double planning_rate = 10;
bool print_timings = false;
bool debug_path_id = false;
double line_check_resolution = 0.1;
double inflation_radius = 0.5;
double inflation_blur_radius = 0.125;
bool publish_viz_paths = false;
int pixels_to_skip = 0;
double open_cv_scale = 0.5;
double h_class_threshold = 0.01;
double min_node_sep_sq = 1.0;
bool publish_all_path_markers = true;
bool visualize_edges = true;
double node_connection_threshold_pix = 1.0;
double collision_threshold = 90;
double trimming_collision_threshold = 60;
double search_radius = 1.5;
double lonely_branch_dist_threshold = 4;
double path_waypoint_sep = 0.1;
bool publish_path_point_markers = false;
double path_vertex_angle_threshold = 45;
double path_vertex_dist_threshold = 0.8;
std::string base_link_frame = "base_link";