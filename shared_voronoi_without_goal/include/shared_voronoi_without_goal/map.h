#ifndef MAP_H_
#define MAP_H_

#include <vector>
#include <string>

/**
 * Same structure as ROS's nav_msgs::OccupancyGrid type
 * Redefined here to decouple from ROS
 **/
class Map
{
public:
	Map() {}
	Map(std::vector<int> in_data, int _width, int _height, double _resolution, std::string _frame_id)
	{
		data.insert(data.begin(), in_data.begin(), in_data.end());
		width = _width;
		height = _height;
		resolution = _resolution;
		frame_id = _frame_id;
	}

	std::vector<signed char> data;
	std::string frame_id;
	double resolution;
	int width;
	int height;
	double robot_radius;

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

	bool worldToMap(double wx, double wy, unsigned int &mx, unsigned int &my) const
	{
		// if (wx < origin.position.x || wy < origin.position.y)
		// 	return false;

		mx = (int)((wx - origin.position.x) / resolution);
		my = (int)((wy - origin.position.y) / resolution);

		if (mx < width && my < height)
			return true;

		return false;
	}

	bool worldToMap(double wx, double wy, double &mx, double &my) const
	{
		// if (wx < origin.position.x || wy < origin.position.y)
		// 	return false;

		mx = (int)((wx - origin.position.x) / resolution);
		my = (int)((wy - origin.position.y) / resolution);

		if (mx < width && my < height)
			return true;

		return false;
	}

	void mapToWorld(double mx, double my, double &wx, double &wy) const
	{
		wx = origin.position.x + mx * resolution;
		wy = origin.position.y + my * resolution;
	}

	signed char getCost(unsigned int mx, unsigned int my) const 
	{
		unsigned int pixel = mx + my * width;
		return data[pixel];
	}
};
#endif