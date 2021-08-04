#ifndef KDTREE_POINT_H_
#define KDTREE_POINT_H_

namespace kdt
{
	class KDTreePoint : public std::array<double, 2>
	{
	public:
		static const int DIM = 2;
		int index;
	};
} // namespace kdt

#endif