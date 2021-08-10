#include <shared_voronoi_without_goal/shared_voronoi_without_goal_node.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <limits>
#include <iostream>
#include <geometry_msgs/Point.h>
#include <actionlib_msgs/GoalID.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <voronoi_msgs_and_types/PathList.h>

bool readParams(ros::NodeHandle &nh)
{
	//Read parameters
	nh.getParam("occupancy_threshold", occupancy_threshold);
	nh.getParam("debug_path_id", debug_path_id);
	nh.getParam("planning_rate", planning_rate);
	nh.getParam("print_timings", print_timings);
	nh.getParam("line_check_resolution", line_check_resolution);
	nh.getParam("pixels_to_skip", pixels_to_skip);
	nh.getParam("open_cv_scale", open_cv_scale);
	nh.getParam("h_class_threshold", h_class_threshold);
	nh.getParam("min_node_sep_sq", min_node_sep_sq);
	nh.getParam("publish_all_path_markers", publish_all_path_markers);
	nh.getParam("visualize_edges", visualize_edges);
	nh.getParam("node_connection_threshold_pix", node_connection_threshold_pix);
	nh.getParam("collision_threshold", collision_threshold);
	nh.getParam("trimming_collision_threshold", trimming_collision_threshold);
	nh.getParam("search_radius", search_radius);
	nh.getParam("lonely_branch_dist_threshold", lonely_branch_dist_threshold);
	nh.getParam("path_waypoint_sep", path_waypoint_sep);
	nh.getParam("publish_path_point_markers", publish_path_point_markers);
	nh.getParam("path_vertex_angle_threshold", path_vertex_angle_threshold);
	nh.getParam("base_link_frame", base_link_frame);
	nh.getParam("inflation_radius", inflation_radius);
	nh.getParam("inflation_blur_radius", inflation_blur_radius);
	nh.getParam("publish_viz_paths", publish_viz_paths);
	nh.getParam("robot_radius", robot_radius);
	nh.getParam("use_elastic_band", use_elastic_band);
	nh.getParam("publish_path_names", publish_path_names);
	nh.getParam("backtrack_plan_threshold", backtrack_plan_threshold);

	//Set parameters for voronoi path object
	v_path.h_class_threshold = h_class_threshold;
	v_path.backtrack_plan_threshold = backtrack_plan_threshold;
	v_path.print_timings = print_timings;
	v_path.node_connection_threshold_pix = node_connection_threshold_pix;
	v_path.min_node_sep_sq = min_node_sep_sq;
	v_path.trimming_collision_threshold = trimming_collision_threshold;
	v_path.search_radius = search_radius;
	v_path.open_cv_scale = open_cv_scale;
	v_path.pixels_to_skip = pixels_to_skip;
	v_path.lonely_branch_dist_threshold = lonely_branch_dist_threshold;
	v_path.path_waypoint_sep = path_waypoint_sep;
	v_path.path_vertex_angle_threshold = path_vertex_angle_threshold;
	v_path.use_elastic_band = use_elastic_band;

	//Elasitc band params
        nh.getParam("num_optim_iterations", num_optim_iterations_);
	nh.getParam("max_recursion_depth_approx_equi", max_recursion_depth_approx_equi_); 
        nh.getParam("internal_force_gain", internal_force_gain_); 
        nh.getParam("external_force_gain", external_force_gain_); 
        nh.getParam("tiny_bubble_distance", tiny_bubble_distance_); 
        nh.getParam("tiny_bubble_expansion", tiny_bubble_expansion_); 
        nh.getParam("min_bubble_overlap", min_bubble_overlap_); 
        nh.getParam("equilibrium_relative_overshoot", equilibrium_relative_overshoot_); 
        nh.getParam("significant_force", significant_force_); 
        nh.getParam("costmap_weight", costmap_weight_); 
        v_path.num_optim_iterations_ = num_optim_iterations_;
        v_path.internal_force_gain_ = internal_force_gain_;
        v_path.external_force_gain_ = external_force_gain_;
        v_path.tiny_bubble_distance_ = tiny_bubble_distance_;
        v_path.tiny_bubble_expansion_ = tiny_bubble_expansion_;
        v_path.min_bubble_overlap_ = min_bubble_overlap_;
        v_path.max_recursion_depth_approx_equi_ = max_recursion_depth_approx_equi_;
        v_path.equilibrium_relative_overshoot_ = equilibrium_relative_overshoot_;
        v_path.significant_force_ = significant_force_;
        v_path.costmap_weight_ = costmap_weight_;
	v_path.updateEBandParams();

	return true;
}

void publishVoronoiViz()
{
	std::vector<GraphNode> nodes;
	std::vector<GraphNode> lonely_nodes;
	std::vector<GraphNode> centers;
	v_path.getObstacleCentroids(centers);
	v_path.getEdges(nodes);
	v_path.getDisconnectedNodes(lonely_nodes);

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
		map.mapToWorld(node.x, node.y, temp_point.x, temp_point.y);

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
	marker_lonely.scale.x = 0.05;
	marker_lonely.scale.y = 0.05;
	marker_lonely.color.a = 0.7;
	marker_lonely.color.r = 1.0;
	marker_lonely.pose.orientation.w = 1.0;
	marker_lonely.points.reserve(lonely_nodes.size());

	for (const auto &node : lonely_nodes)
	{
		geometry_msgs::Point temp_point;
		map.mapToWorld(node.x, node.y, temp_point.x, temp_point.y);

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
		map.mapToWorld(center.x, center.y, temp_point.x, temp_point.y);

		marker_obstacles.points.push_back(std::move(temp_point));
	}

	marker_array.markers.push_back(std::move(marker_obstacles));
	marker_array.markers.push_back(std::move(marker));
	marker_array.markers.push_back(std::move(marker_lonely));
	edges_viz_pub.publish(marker_array);
}

//Map subscriber
void globalMapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
	//Copy all required data to internal map storage
	map.height = msg->info.height;
	map.width = msg->info.width;
	map.frame_id = msg->header.frame_id;
	map.resolution = msg->info.resolution;
	map.origin.position.x = msg->info.origin.position.x;
	map.origin.position.y = msg->info.origin.position.y;

	//Dilate image
	cv::Mat cv_map = cv::Mat(msg->data).reshape(0, msg->info.height);
	cv_map.convertTo(cv_map, CV_8UC1);

	//Dilation size on one side of obstacle is pixel/map_resolution/2
	//ie 20 pixels * 0.05m/px / 2 = 0.5m inflation on obstacles
	int dilate_size = inflation_radius * 2 / map.resolution;
	cv::Mat structure_elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_size, dilate_size));
	cv::dilate(cv_map, cv_map, structure_elem);

	//Blur. Blur kernel must be odd number
	int blur_size = inflation_blur_radius * 2 / map.resolution;
	blur_size += (blur_size + 1) % 2;

	cv::GaussianBlur(cv_map, cv_map, cv::Size(blur_size, blur_size), 3);
	if (cv_map.isContinuous())
		map.data.assign(cv_map.data, cv_map.data + cv_map.total() * cv_map.channels());

	else
	{
		for (int i = 0; i < cv_map.rows; ++i)
			map.data.insert(map.data.end(), cv_map.ptr<uchar>(i), cv_map.ptr<uchar>(i) + cv_map.cols * cv_map.channels());
	}

	nav_msgs::OccupancyGrid map_dilated = *msg;
	map_dilated.data = map.data;
	v_path.mapToGraph(&map);
	map_pub.publish(map_dilated);

	if (visualize_edges)
		publishVoronoiViz();
}

void cancelCB(const actionlib_msgs::GoalIDConstPtr &msg)
{
	v_path.clearPreviousPaths();
}

void makePlan(const ros::WallTimerEvent &e)
{
	//Map is not ready, return
	if (map.frame_id == "")
		return;

	//Get robot position
	geometry_msgs::TransformStamped base_link_to_map_tf;
	geometry_msgs::Pose robot_pose;
	try
	{
		base_link_to_map_tf = tf_buffer.lookupTransform(map.frame_id, base_link_frame, ros::Time(0), ros::Duration(1.0/planning_rate));
		tf2::doTransform<geometry_msgs::Pose>(robot_pose, robot_pose, base_link_to_map_tf);
	}
	catch (tf2::TransformException &Exception)
	{
		ROS_ERROR_STREAM(Exception.what());
	}

	GraphNode start_point;
	map.worldToMap(robot_pose.position.x, robot_pose.position.y, start_point.x, start_point.y);

	//DFS planning
	all_paths = v_path.backtrackPlan(start_point);
	if(publish_viz_paths)
	{
		all_paths = v_path.getVizPaths();
	}

	if(debug_path_id)
	{
		for(int i = 0; i < all_paths.size(); ++i)
			std::cout << "Path " << i << " id " << all_paths[i].id << "\n";
	}

	//If paths are found
	if (!all_paths.empty())
	{
		//Interpolate path to get even separation between waypoints
		v_path.interpolatePaths(all_paths, path_waypoint_sep);
		std::vector<std::vector<geometry_msgs::PoseStamped>> all_paths_meters;
		all_paths_meters.resize(all_paths.size());
		visualization_msgs::MarkerArray marker_array;

		//Convert node numbers to position on map for path
		std_msgs::Header header;
		header.stamp = ros::Time::now();
		header.frame_id = map.frame_id;
		double marker_lifetime = 2.0 / planning_rate;
		for (int i = 0; i < all_paths.size(); ++i)
		{
			visualization_msgs::Marker marker, points_marker, path_number;
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
				marker.lifetime = ros::Duration(marker_lifetime);

				path_number.header = header;
				path_number.id = i + 2 * all_paths.size();
				path_number.ns = std::string("Path ") + std::to_string(i) + std::string(" text");
				path_number.action = 0;
				path_number.type = 9;
				path_number.scale.z = 0.5;
				path_number.color.r = 0.0;
				path_number.color.g = 0.0;
				path_number.color.b = 0.0;
				path_number.color.a = 1.0;
				path_number.lifetime = ros::Duration(marker_lifetime);
				path_number.text = "Path " + std::to_string(all_paths[i].id);

				if (publish_path_point_markers)
				{
					points_marker.header = header;
					points_marker.ns = std::string("Path Points ") + std::to_string(i);
					points_marker.id = i + all_paths.size();
					points_marker.type = 8;
					points_marker.action = 0;
					points_marker.scale.x = 0.15;
					points_marker.scale.y = 0.15;
					points_marker.color.g = 1.0;
					points_marker.color.a = 0.8;
					points_marker.pose.orientation.w = 1.0;
					points_marker.lifetime = ros::Duration(marker_lifetime);
				}
			}

			//Loop through all the nodes for path i to convert to meters and generate visualization markers if enabled
			for (int k = 0; k < all_paths[i].path.size(); ++k)
			{
				const auto &pose = all_paths[i].path[k];
				geometry_msgs::PoseStamped new_pose;
				new_pose.header = header;
				map.mapToWorld(pose.x, pose.y, new_pose.pose.position.x, new_pose.pose.position.y);
				new_pose.pose.position.z = 0;

				//TODO: Set orientation of intermediate poses
				new_pose.pose.orientation.w = 1;

				//Add path number halfway through the path
				if (k == all_paths[i].path.size() / 2)
					path_number.pose = new_pose.pose;

				all_paths_meters[i].push_back(std::move(new_pose));

				if (publish_all_path_markers)
				{
					marker.points.push_back(new_pose.pose.position);

					if (publish_path_point_markers)
						points_marker.points.push_back(new_pose.pose.position);
				}
			}

			if (publish_all_path_markers)
			{
				marker_array.markers.push_back(marker);

				if(publish_path_names)
					marker_array.markers.push_back(path_number);

				if (publish_path_point_markers)
					marker_array.markers.push_back(points_marker);
			}

			//Adjust orientation of start and end positions
			if (!all_paths_meters[i].empty())
				all_paths_meters[i][0].pose.orientation = robot_pose.orientation;
		}

		//Publish visualization for all available paths
		if (publish_all_path_markers)
			all_paths_pub.publish(marker_array);

		//Publish all generated paths
		voronoi_msgs_and_types::PathList path_list;
		for (int i = 0; i < all_paths_meters.size(); ++i)
		{
			nav_msgs::Path temp_path;
			temp_path.header.stamp = ros::Time::now();
			temp_path.header.frame_id = map.frame_id;
			temp_path.header.seq = all_paths[i].id;
			temp_path.poses = all_paths_meters[i];
			path_list.paths.emplace_back(std::move(temp_path));
		}
		all_paths_ind_pub.publish(path_list);
	}

	return;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "shared_voronoi_without_goal");
	ros::NodeHandle nh("~");

	//Listener has to be declared and initialized after ros::init
	tf2_ros::TransformListener tf_listener(tf_buffer);

	readParams(nh);

	//Subscribe and advertise related topics
	global_map_sub = nh.subscribe("/map", 1, &globalMapCB);

	// if (!static_global_map)
	// 	global_update_sub = nh_private.subscribe("global_costmap/costmap_updates", 1, &SharedVoronoiGlobalPlanner::globalCostmapUpdateCB, this);

	// if (subscribe_local_costmap)
	// 	local_costmap_sub = nh_private.subscribe("local_costmap/costmap", 1, &SharedVoronoiGlobalPlanner::localCostmapCB, this);

	// //Subscribe to joystick output to get direction selected by user
	// user_vel_sub = nh.subscribe(joystick_topic, 1, &SharedVoronoiGlobalPlanner::cmdVelCB, this);

	// //Subscribe to odometry to make sure that sorted node list is updated
	// odom_sub = nh.subscribe(odom_topic, 1, &SharedVoronoiGlobalPlanner::odomCB, this);

	// //Subscribe to preferred path from belief update
	// preferred_path_sub = nh.subscribe("preferred_path_ind", 1, &SharedVoronoiGlobalPlanner::preferredPathCB, this);

	//Subscribe to move_base cancel if published
	move_base_cancel_sub = nh.subscribe("/move_base/cancel", 1, &cancelCB);

	//Visualization topics
	all_paths_pub = nh.advertise<visualization_msgs::MarkerArray>("all_paths_viz", 1);
	// user_direction_pub = nh.advertise<visualization_msgs::Marker>("user_direction_viz", 1);
	edges_viz_pub = nh.advertise<visualization_msgs::MarkerArray>("voronoi_edges_viz", 1, true);

	//Plan and voronoi diagram related topics
	map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/map_dilated", 1);
	// global_path_pub = nh.advertise<nav_msgs::Path>("plan", 1);
	// adjacency_list_pub = nh.advertise<shared_voronoi_global_planner::AdjacencyList>("adjacency_list", 1, true);
	// node_info_pub = nh.advertise<shared_voronoi_global_planner::NodeInfoList>("node_info", 1, true);
	// sorted_nodes_pub = nh.advertise<shared_voronoi_global_planner::SortedNodesList>("sorted_nodes", 1, true);
	all_paths_ind_pub = nh.advertise<voronoi_msgs_and_types::PathList>("all_paths", 1);

	//Create timer to update Voronoi diagram, use one shot timer if update rate is 0
	if (planning_rate != 0)
		voronoi_update_timer = nh.createWallTimer(ros::WallDuration(1.0 / planning_rate), &makePlan);
	else
	{
		ROS_ERROR("Update rate should be greater than 0!");
		return -1;
	}

	ros::spin();

	return 0;
}
