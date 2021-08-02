#ifndef GRAPH_NODE_H_
#define GRAPH_NODE_H_

#include <map>
#include <cmath>

/**
 * Type used to store coordinates of nodes. Coordinates are pixels in the map
 **/
class GraphNode
{
public:
	GraphNode() : x(0), y(0) {}
	GraphNode(double _x, double _y) : x(_x), y(_y) {}
	GraphNode(std::pair<double, double> in_pair) : x(in_pair.first), y(in_pair.second) {}
	double x;
	double y;

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
#endif