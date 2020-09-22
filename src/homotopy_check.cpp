#include <iostream>
#include <complex>
#include <vector>

std::vector<std::vector<std::complex<double>>> complex_path;
std::vector<std::complex<double>> obstacles;
std::vector<std::complex<double>> homotopy_class;

std::complex<double> BL(0, 0);
std::complex<double> TR(11, 15);
std::complex<double> f_not(const std::complex<double> &z, const int &n)
{
    //a - b = 0
    //a + b = n - 1
    double a = (n - 1) / 2.0;
    double b = a;
    return std::pow((z - BL), a) + std::pow((z - TR), b);
}

int main(int argc, char **argv)
{
    //Insert path 1
    complex_path.push_back(std::vector<std::complex<double>>());
    complex_path[0].emplace_back(8, 2);
    complex_path[0].emplace_back(7, 2);
    complex_path[0].emplace_back(6, 2);
    complex_path[0].emplace_back(6, 3);
    complex_path[0].emplace_back(6, 4);
    complex_path[0].emplace_back(6, 5);
    complex_path[0].emplace_back(6, 6);
    complex_path[0].emplace_back(6, 7);
    complex_path[0].emplace_back(6, 8);
    complex_path[0].emplace_back(6, 9);
    complex_path[0].emplace_back(6, 10);
    complex_path[0].emplace_back(6, 11);
    complex_path[0].emplace_back(6, 12);
    complex_path[0].emplace_back(6, 13);
    complex_path[0].emplace_back(7, 13);
    complex_path[0].emplace_back(8, 13);

    //Insert path 2
    complex_path.push_back(std::vector<std::complex<double>>());
    complex_path[1].emplace_back(8, 2);
    complex_path[1].emplace_back(9, 2);
    complex_path[1].emplace_back(10, 2);
    complex_path[1].emplace_back(10, 3);
    complex_path[1].emplace_back(10, 4);
    complex_path[1].emplace_back(10, 5);
    complex_path[1].emplace_back(10, 6);
    complex_path[1].emplace_back(10, 7);
    complex_path[1].emplace_back(10, 8);
    complex_path[1].emplace_back(10, 9);
    complex_path[1].emplace_back(10, 10);
    complex_path[1].emplace_back(10, 11);
    complex_path[1].emplace_back(10, 12);
    complex_path[1].emplace_back(10, 13);
    complex_path[1].emplace_back(9, 13);
    complex_path[1].emplace_back(8, 13);

    //Insert path 3
    complex_path.push_back(std::vector<std::complex<double>>());
    complex_path[2].emplace_back(8, 2);
    complex_path[2].emplace_back(7, 2);
    complex_path[2].emplace_back(6, 2);
    complex_path[2].emplace_back(5, 2);
    complex_path[2].emplace_back(4, 2);
    complex_path[2].emplace_back(4, 3);
    complex_path[2].emplace_back(4, 4);
    complex_path[2].emplace_back(4, 5);
    complex_path[2].emplace_back(4, 6);
    complex_path[2].emplace_back(4, 7);
    complex_path[2].emplace_back(4, 8);
    complex_path[2].emplace_back(4, 9);
    complex_path[2].emplace_back(4, 10);
    complex_path[2].emplace_back(4, 11);
    complex_path[2].emplace_back(4, 12);
    complex_path[2].emplace_back(4, 13);
    complex_path[2].emplace_back(5, 13);
    complex_path[2].emplace_back(6, 13);
    complex_path[2].emplace_back(7, 13);
    complex_path[2].emplace_back(8, 13);

    //Insert obstacle 1
    obstacles.emplace_back(8, 8);

    //Insert obstacle 2
    obstacles.emplace_back(8, 5);

    obstacles.emplace_back(2, 4);

    homotopy_class = std::vector<std::complex<double>>(complex_path.size());

    //Calculate homotopy class for each path
    for (int j = 0; j < complex_path.size(); ++j)
    {
        std::vector<std::complex<double>> path = complex_path[j];
        std::complex<double> path_sum;

        //Go through each edge of the path
        for (int i = 1; i < path.size(); ++i)
        {
            std::complex<double> edge_sum(0, 0);
            //Each edge must iterate through all obstacles
            for (auto obs : obstacles)
            {
                //Calculate Al value, initialize with 1,1
                std::complex<double> al_denom_prod(1,1);
                for(auto excl_obs : obstacles)
                {
                    if(excl_obs == obs)
                        continue;

                    al_denom_prod *= (obs - excl_obs);
                }
                std::complex<double> al = f_not(obs, obstacles.size()) / al_denom_prod;

                double real_part = std::log(std::abs(path[i] - obs)) - std::log(std::abs(path[i - 1] - obs));
                double im_part = std::arg(path[i] - obs) - std::arg(path[i - 1] - obs);

                //Get smallest angle
                while (im_part > M_PI)
                    im_part -= 2 * M_PI;

                while (im_part < -M_PI)
                    im_part += 2 * M_PI;

                edge_sum += (std::complex<double>(real_part, im_part) * al);
            }

            //Add this edge's sum to the path sum
            path_sum += edge_sum;
        }

        homotopy_class[j] = path_sum;
    }

    std::cout << "Class 1: " << homotopy_class[0].real() << " + " << homotopy_class[0].imag() << "i\n";
    std::cout << "Class 2: " << homotopy_class[1].real() << " + " << homotopy_class[1].imag() << "i\n";
    std::cout << "Class 3: " << homotopy_class[2].real() << " + " << homotopy_class[2].imag() << "i\n";

    return 0;
}