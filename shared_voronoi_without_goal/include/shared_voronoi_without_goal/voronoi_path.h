#ifndef VORONOI_PATH_H
#define VORONOI_PATH_H

#define JC_VORONOI_IMPLEMENTATION

//Uncomment to change voronoi calculations from floating point to double floating point
#define JCV_REAL_TYPE double
#define JCV_ATAN2 atan2
#define JCV_SQRT sqrt
#define JCV_FLT_MAX 1.7976931348623157E+308
#define JCV_PI 3.141592653589793115997963468544185161590576171875

#include "jc_voronoi_clip.h"
#include <chrono>
#include <limits>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <complex>
#include <mutex>
#include <memory>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace voronoi_path
{
    /**
     * Type used to store coordinates of nodes. Coordinates are pixels in the map
     **/
    struct GraphNode
    {
        double x;
        double y;
        GraphNode() : x(0), y(0) {}
        GraphNode(double _x, double _y) : x(_x), y(_y) {}
        GraphNode(std::pair<double, double> in_pair) : x(in_pair.first), y(in_pair.second) {}

        GraphNode operator*(const double &mult) const
        {
            return GraphNode(x * mult, y * mult);
        }

        GraphNode operator/(const double &mult) const
        {
            return GraphNode(x / mult, y / mult);
        }

        GraphNode operator+(const double &incr) const
        {
            return GraphNode(x + incr, y + incr);
        }

        GraphNode operator-(const double &incr) const
        {
            return GraphNode(x - incr, y - incr);
        }

        GraphNode &operator+=(const GraphNode &incr)
        {
            x = x + incr.x;
            y = y + incr.y;
            return *this;
        }

        GraphNode operator+(const GraphNode &incr) const
        {
            return GraphNode(x + incr.x, y + incr.y);
        }

        GraphNode operator-(const GraphNode &incr) const
        {
            return GraphNode(x - incr.x, y - incr.y);
        }

        bool operator==(const GraphNode &rhs) const
        {
            return (x == rhs.x && y == rhs.y);
        }

        double getMagnitude() const
        {
            return sqrt(pow(x, 2) + pow(y, 2));
        }

        void setUnitVector()
        {
            double magnitude = sqrt(pow(x, 2) + pow(y, 2));
            x /= magnitude;
            y /= magnitude;
        }
    };

    /**
     * Struct to store vector of GraphNodes (a path) and its corresponding id
     **/
    struct Path
    {
        uint32_t id;
        std::vector<GraphNode> path;

        //Constructors
        Path() : id(-1){}
        Path(uint32_t _id, std::vector<GraphNode> && in_path)
        {
            id = _id;
            path = in_path;
        }
    };

    /**
     * Same structure as ROS's nav_msgs::OccupancyGrid type
     * Redefined here to decouple from ROS
     **/
    struct Map
    {
        std::vector<signed char> data;
        std::string frame_id;
        double resolution;
        int width;
        int height;
        struct
        {
            struct
            {
                double x;
                double y;
                double z;
            } position;

            struct
            {
                double x;
                double y;
                double z;
                double w;
            } orientation;
        } origin;

        Map() {}
        Map(std::vector<int> in_data, int _width, int _height, double _resolution, std::string _frame_id)
        {
            data.insert(data.begin(), in_data.begin(), in_data.end());
            width = _width;
            height = _height;
            resolution = _resolution;
            frame_id = _frame_id;
        }
    };

    /**
     * Structure definition for use during A* search algorithm only
     * Contains the information required for path searching
     **/
    struct NodeInfo
    {
        double cost_upto_here;
        double cost_to_goal;
        double total_cost;

        void updateCost()
        {
            total_cost = cost_upto_here + cost_to_goal;
        }
    };

    class VoronoiPath
    {
    public:
        VoronoiPath();

        /**
         * Get all the nodes in the voronoi diagram. ith node is connected to i+1 node, i is even.
         * @param edges vector passed by reference to store the edges
         * @return boolean indicating success
         **/
        bool getEdges(std::vector<GraphNode> &edges);

        /**
         * Get nodes which are singly connected (only has 1 edge connected to it)
         * @param nodes vector passed by reference to store the nodes
         * @return boolean indicating success
         **/
        bool getDisconnectedNodes(std::vector<GraphNode> &nodes);

        /**
         * Get centroids of obstacles 
         * @param centroids vector passed by reference to store the centroids
         * @return boolean indicating success
         **/
        bool getObstacleCentroids(std::vector<GraphNode> &centroids);

        /**
         * Get the costs of all paths 
         * @return vector containing all the costs
         **/
        std::vector<double> getAllPathCosts();

        /**
         * Gets voronoi graph from map_
         * @param map_ map to use for generation of voronoi graph
         * @return boolean indicating success
         **/
        bool mapToGraph(Map* map_ptr_);

        /**
         * Get adjacency list of the current voronoi graph
         * @return adjacency list
         **/
        std::vector<std::vector<int>> getAdjList();

        /**
         * Get the corresponding node coordinates of current voronoi_graph. Coordinates are in meters
         * @return vector containing coordinates of voronoi nodes
         **/
        std::vector<GraphNode> getNodeInfo();

        /**
         * Returns a list of voronoi nodes sorted by distance from the current position of robot
         * @return vector of voronoi nodes sorted by ascending distance from current robot position
         **/
        std::vector<std::pair<double, int>> getSortedNodeList(GraphNode position);

        /**
         * Debugging method used to print all edges generated for voronoi graph
         **/
        void printEdges();

        /**
         * Method called to find num_paths shortest paths using A* and Yen's algorithm, with homotopy class checking
         * @param start start position, in pixels wrt global map origin
         * @param end end position, in pixels wrt to global map origin
         * @param num_paths total number of paths to find. ie 2 will return the 2 (most likely) shortest paths
         * @return vector containing all the paths found
         **/
        std::vector<Path> getPath(const GraphNode &start, const GraphNode &end, const int &num_paths);

        /**
         * Alternate planning method using robot's current position and backtracking to find all paths that emanate from robot position.
         * Path lengths are limited to certain threshold
         * @param robot_pos robot's current position
         * @return vector containing all paths found
         **/
        std::vector<Path> backtrackPlan(const GraphNode &start);

        /**
         * Replan based on paths generated in the previous time step
         **/
        std::vector<Path> replan(GraphNode &start, GraphNode &end, int num_paths, int &pref_path);

        /**
         * Set the location of local vertices. Vertices are in pixels, in global map's frame
         * @param vertices vector containing the 4 corners of local costmap
         **/
        void setLocalVertices(const std::vector<GraphNode> &vertices);

        /**
         * Interpolate paths give a certain threshold distance
         * @param paths vector of paths to interpolate
         * @param path_waypoint_sep min separation between waypoints on the path
         * @return bool indicating success
         **/
        bool interpolatePaths(std::vector<Path> &paths, double path_waypoint_sep);

        /**
         * Clear the vector storing all previous paths
         * @return boolean indicating success
         **/
        bool clearPreviousPaths();

        /**
         * Check if there are previous paths
         * @return boolean indicating result
         **/
        bool hasPreviousPaths();

        /**
         * Return paths that are used for debugging/visualization. Shows which path the current path is connected to
         * @return vector of all viz paths
         **/
        std::vector<Path> getVizPaths();

        /**
         * Pixel resolution to increment when checking if an edge collision occurs. Value of 0.1 means the edge will
         * be checked at every 0.1 pixel intervals
         **/
        double line_check_resolution = 0.1;

        /**
         * Set print_timings to print all timings for critical sections of the code. Used for debugging
         **/
        bool print_timings = false;

        /**
         * Threshold before a pixel is considered occupied. If pixel value is < occupancy_threshold, it is considered free
         **/
        int occupancy_threshold = 100;

        /**
         * Threshold before a pixel is considered occupied during collision checking, this is same as occupancy_threshold but 
         * collision_threshold is used when checking if an edge collides with obstacles. Can be used in conjunction with
         * ROS's costmap inflation to prevent planner from planning in between narrow spaces
         **/
        int collision_threshold = 85;

        /**
         * Threshold used for trimming paths, should be smaller than collision_threshold to prevent robot from getting stuck
         **/
        int trimming_collision_threshold = 75;

        /**
         * Radius to search around robot location to try and find an empty cell to connect to start of previous path, meters
         **/
        double search_radius = 1.5;
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
         * Percentage threshold to classify a homotopy class as same or different. Ideally, same homotopy classes should have identical 
         * complex values, but since "double" representation is used, some difference might be present for same homotopy classes
         * 0.01 means 1 percent difference in class values
         **/
        double h_class_threshold = 0.01;

        /**
         * Minimum separation between nodes. If nodes are less than this value (m) apart, they will be cleaned up
         **/
        double min_node_sep_sq = 1.0;

        /**
         * Distance to put the extra point which is used to ensure continuity, units meter
         **/
        double extra_point_distance = 1.0;

        /**
         * If there is a node within this threshold away from a node that only has 1 connection, they will both be connected
         **/
        int node_connection_threshold_pix = 1;

        /**
         * Meters squared along a lonely branch to travel before aborting. If lonely branch is too long, it will not be pruned
         **/
        double lonely_branch_dist_threshold = 4;

        /**
         * Minimum distance between poses on global path generated
         **/
        double path_waypoint_sep = 0.2;

        /**
         * Degrees angle threshold before 3 points on path are considered stuck, causing center point to be removed
         **/
        double path_vertex_angle_threshold = 25;

        /**
         * Distance in meters before exhaustive path exploration terminates
         **/
        double backtrack_plan_threshold = 5;

        /**
         * Cycle check node depth threshold
         **/
        int node_depth_threshold = 5; 

        /**
         * Threshold distance in meters for last branch in order for deadend path to be considered
         **/
        double last_branch_dist_thresh = 2.5;

    private:
        /**
         * Pointer to map from the ROS side of planner
         **/
        Map* map_ptr;

        /**
         * Previous time steps map origin information, used to realign previous time step's paths
         **/
        Map prev_map;

        /**
         * Vector storing sorted list of voronoi nodes square distances from the current robot position
         **/
        std::vector<std::pair<double, int>> sorted_node_list;

        /**
         * Adjacency list retrieved from the edge vector. Each i in adj_list[i] is a vector of nodes that are connected to node i
         * ie. adj_list[i][j] is connected to node i
         **/
        std::vector<std::vector<int>> adj_list;

        /**
         * Vector storing pixel coordinates of all nodes. Index refers to the node number
         * ie. Node i is at (node_inf[i].x, node_inf[i].y) coordinate
         **/
        std::vector<GraphNode> node_inf;

        /**
         * Vector storing the coordinate of the 4 corners of the local costmap (ROS). Used to unsure that a path can be found 
         * even in a sparse global map
         **/
        std::vector<GraphNode> local_vertices;

        /**
         * Centers of centroids in complex form, for use when calculating homotopy classes only
         **/
        std::vector<std::complex<double>> centers;

        /**
         * Bottom left of current global map, for use during calculating homotopy classes
         **/
        std::complex<double> BL = std::complex<double>(0, 0);

        /**
         *  Top right of current global map, for use during calculating homotopy classes
         **/
        std::complex<double> TR = std::complex<double>(1, 1);

        /**
         * Mutex to lock access for adj_list 
         **/
        std::mutex voronoi_mtx;

        /**
         * Precomputed coefficients for all obstacle centroids for use in calculating homotopy classes
         **/
        std::vector<std::complex<double>> obs_coeff;

        /**
         * Stores total number of nodes, ie adj_list.size()
         **/
        int num_nodes = 0;

        /**
         * Max number of nodes that can be used to generate a bezier subsection. 26 choose 13 is 10400600. Higher
         * values increases the likelihood of integer overflow
         **/
        int bezier_max_n = 26;

        /**
         * Vector to store all previously found paths for maintaining and trimming
         **/
        std::vector<Path> previous_paths;

        /**
         * Vector to store all previously found viz paths
         **/
        std::vector<Path> viz_paths;

        /**
         * Vector storing all the costs of previous paths
         **/
        std::vector<double> previous_path_costs;

        //******************* Methods *******************/

        /**
         * Returns a unique ID for labelling paths generated since the start of the program.
         * It is a monotically increasing counter
         * @return uint32_t storing the ID
         **/
        uint32_t getUniqueID();

        /**
         * Find centroid of obstacles using opencv findContour(), uses data from 'map' variable
         * @return vector of complex numbers representing coordinates of centroids, for use in homotopy class checking
         **/
        std::vector<std::complex<double>> findObstacleCentroids();

        /**
         * Method for threading the process of filling up occupancy vector by iterating through the map
         * @param start_index pixel to start looping from
         * @param num_pixels number of pixels to iterate over and check if it's occupied
         * @return returns vector of coordinates to occupied pixels
         **/
        std::vector<jcv_point> fillOccupancyVector(const int &start_index, const int &num_pixels);

        /**
         * Hashing function for 2 doubles, order of values matters
         * @param x first value
         * @param y second value
         * @return hash
         **/
        uint32_t hash(const double &x, const double &y);

        /**
         * Find nearest starting and ending node, given starting and ending coordinates
         * @param start starting coordinates in pixels
         * @param end ending coordinates in pixels
         * @param start_node node number corresponding to nearest node in adj_list
         * @param end_node node number corresponding to nearest node in adj_list
         * @return boolean indicating success
         **/
        bool getNearestNode(const GraphNode &start, const GraphNode &end, int &start_node, int &end_node);

        /**
         * Find kth shortest paths using Yen's algorithm
         * @param start_node node number of starting node
         * @param end_node node number of ending node
         * @param shortestPath shortest path that was previously found using findShortestPath()
         * @param all_paths all kth shortest paths
         * @param num_paths number of paths to find. If num_paths = 1, total paths including shortest path will be 2
         * @return boolean indicating success
         **/
        bool kthShortestPaths(const int &start_node, const int &end_node, const std::vector<int> &shortestPath, std::vector<std::vector<int>> &all_paths, const int &num_paths);

        /**
         * Find shortest path using A* algorithm and Euclidean distance heuristic
         * @param start_node node number of starting node
         * @param end_node node number of ending node
         * @param path shortest path that was found
         * @param cost cost of the shortest path, sum of all edge lengths in the path, units are pixels
         * @return boolean indicating success
         **/
        bool findShortestPath(const int &start_node, const int &end_node, std::vector<int> &path);

        /**
         * Removes voronoi vertices that are in obstacles
         **/
        void removeObstacleVertices(std::vector<const jcv_edge*>& edge_vector);

        /**
         * Removes edges that collide with obstacles
         **/
        void removeCollisionEdges(std::vector<const jcv_edge*>& edge_vector);

        /**
         * Calculate the minimum angle between 2 vectors
         * @param vec1 2 element vector (x,y)
         * @param vec2 2 element vector (x,y)
         * @return minimum angle between vec1 and vec2 in radians
         **/
        double vectorAngle(const double vec1[2], const double vec2[2]);

        /**
         * Checks if an edge connection start and end collides with anything in map
         * @param start pixel position of start node
         * @param end pixel position of end node
         * @return returns true if edge connecting start to end collides with obstacles
         **/
        bool edgeCollides(const GraphNode &start, const GraphNode &end, int threshold);

        /**
         * Manhattan distance from a to b
         * @param a pixel position of point a
         * @param b pixel position of point b
         * @return distance from a to b using Manhattan distance formula
         **/
        double manhattanDist(const GraphNode &a, const GraphNode &b);

        /**
         * Euclidean distance from a to b
         * @param a pixel position of point a
         * @param b pixel position of point b
         * @return distance from a to b using Euclidean distance formula (hypotenuse of a right angle triangle)
         **/
        double euclideanDist(const GraphNode &a, const GraphNode &b);

        /**
         * Get the number of nodes currently in adjacency list
         * @return number of nodes in adjacency list
         **/
        int getNumberOfNodes();

        /**
         * Calculate the homotopy class, algorithm from paper "Search-Based Path Planning with Homotopy Class Constraints"
         * by Subhrajit Bhattacharya et al https://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-216.pdf
         * @param path_ path to calculate homotopy class
         * @return complex value representing the homotopy class of path_
         **/
        std::complex<double> calcHomotopyClass(const std::vector<GraphNode> &path_);

        /**
         * Convert node based path to pixel based path
         * @param path_ path to be converted, will not be modified
         * @return Vector containing pixels of the path
         **/
        std::vector<GraphNode> convertToPixelPath(const std::vector<int> &path_);

        /**
         * Trim the beginning of path such that the starting node is directly connected to the node before X,
         * where X is the node that will cause a collision if the starting node is directly connected to
         * @param path path to trim
         * @return bool indicating success
         **/
        bool contractPath(std::vector<GraphNode> &path);

        /**
         * Find if any of the vertices are in concave area of costmap, causing the path to become stuck
         * Remove the middle vertex if found
         * @param path path to check for stuck vertex
         * @return bool indicating success
         **/
        bool findStuckVertex(std::vector<GraphNode> &path);

        /** 
         * Interpolate paths before contraction to allow smoother result, uses the method above to achieve contraction
         * @param paths paths to contract
         * @return bool indicating success
         **/
        bool interpolateContractPaths(std::vector<Path> &paths);

        /**
         * Checks if the two complex homotopy classes are outside the threshold, ie unique/different
         * @param complex_1 class 1 to check
         * @param complex_2 class 2 to check
         * @return bool indicating true if the 2 classes are unique
         **/
        bool isClassDifferent(const std::complex<double> &complex_1, const std::complex<double> &complex_2);

        /**
         * Traverses recursively a branch from dead end side towards branch side, if branch is found, then entire series of edges are removed
         * If branch is not found before lonely_branch_dist_threshold, then branch is not removed
         * @param curr_node initialize this witht the lonely node, the node at the dead end
         * @param prev_node initialize with -1
         * @param cum_dist initialize with 0
         * @return bool indicating whether or not to prune this branch. Used in backtracking, and for notification only. No manual deletion needed
         **/
        bool removeExcessBranch(std::vector<std::vector<int>> &new_adj_list, double thresh, int curr_node, int prev_node = -1, double cum_dist = 0);        

        /**
         * Convert voronoi edges into an adjacency list used for path finding
         * @param edge_vector the original edge_vector used to generate adjacency list
         * @return boolean indication success
         **/
        bool edgesToAdjacency(const std::vector<const jcv_edge*>& edge_vector);

        /**
         * Check if a point lies in a square formed by points on 2 opposite vertices
         * @param point point to check
         * @param line_point_a first point of the line
         * @param line_point_b second point of the line, order of points do not matter
         * @return boolean indicating whether the point lies on the line
         **/
        bool liesInSquare(const GraphNode &point, const GraphNode &line_point_a, const GraphNode &line_point_b);

        /**
         * Check if there is a cycle within certain depth from a given starting node
         * @param start starting node index
         * @param depth depth before terminating cycle search
         * @return bool indicating whether there is a cycle
         **/
        bool hasCycle(int cur_node, int cur_depth, std::vector<int> &visited_list, int prev = -1);

        /**
         * Checks if 2 paths from backtrack planning are distinct. Combines the endpoints of both paths with a interpolated curve
         * and then uses homotopy check to see if enclosed area contains obstacle point, if yes then they are distinct.
         * If interpolated curve intersects an obstacle, then the paths are also distinct.
         * @param path1 iterator to first path to check
         * @param path2 iterator to second path to check
         * @return boolean indicating true if distinct
         **/
        bool isBacktrackDistinct(std::vector<Path>::iterator &path1, std::vector<Path>::iterator &path2);

        /**
         * Link a vector of paths from current time step to paths from previous time step. If 2 paths are linked, then the path from current
         * time step will inherit the path ID from the linked path of previous time step.
         * Linking logic makes use of isBacktrackDistinct
         * @param prev_paths vector to all paths from previous time step
         * @param cur_paths vector to all paths from current time step
         * @return return a modified version of cur_paths which visualizes the paths that they were connected to. Used for visualizing/debugging
         **/
        std::vector<Path> linkBacktrackPaths(std::vector<Path> &prev_paths, std::vector<Path> &cur_paths);

        /**
         * Returns a vector pair of each path and its corresponding average heading
         * @param paths paths to compute average headings
         * @return vector of pair <average_heading, path index in array>
         **/
        std::vector<std::pair<double, int>> getPathHeadings(const std::vector<Path> &paths);

        /**
         * Template method to flip any vector pair. From vector<pair<A, B>> to vector<pair<B, A>>
         * @param vec original vector pair of <A, B>
         * @return swapped vector pair of <B, A>
         **/
        template <class A, class B>
        std::vector<std::pair<B, A>> flipPairVector(std::vector<std::pair<A, B>> &vec) 
        {
            std::vector<std::pair<B, A>> ret;
            for (const auto elem : vec)
                ret.emplace_back(elem.second, elem.first);

            return ret;
        }

        /**
         * Helper function for public method backtrackPlan
         * Uses dfs and backtracking to find unique paths diverging out from robot position
         * @param path path vector for appending or removing nodes
         * @param cur_dist current distance from starting position
         * @param last_branch_dist distance to last branch node from current position
         * @param prev_node previous node, ie "where I came from"
         * @param cur_node current node number
         * @param paths vector of all paths. Appended to when the termination condition is met
         * @param backtrack_plan_threshold distance from start position to terminate the dfs planning
         **/
        void backtrack(std::vector<int> &path, double cur_dist, double last_branch_dist, const int &prev_node, const int &cur_node, std::vector<std::vector<int>> &paths, const double &backtrack_plan_threshold);
    };

} // namespace voronoi_path

#endif