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
};
#endif