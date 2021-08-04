#ifndef KDTREE_POINT_H_
#define KDTREE_POINT_H_

#include <vector>

namespace kdt
{
	class KDTreePoint
	{
	public:
		KDTreePoint(const double &x, const double &y)
		{
			points.resize(2);
			points[0] = x;
			points[1] = y;
		};

		double &operator[](int index)
		{
			return points[index];
		}

		const double &operator[](int index) const
		{
			return points[index];
		}

		std::vector<double> points;
		static const int DIM = 2;
		int index;
	};
} // namespace kdt

#endif