#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>
#include <map>
#include <vector>

#include "adaboost.hpp"
#include "feature.hpp"
#include "label.hpp"
#include "loaddata.hpp"
#include "matplotlibcpp.hpp"
#include "normalize.hpp"
#include "utile.hpp"
namespace plt = matplotlibcpp;

int main() {
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> data = LoadData::LoadData("../xy_data.txt");
    for (std::size_t frame_id = 0; frame_id < data.size(); frame_id++) {
        const Eigen::MatrixXd &d = data[frame_id];
        std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Seg_data = Utile::Segment(d);
        std::cout << "There are " << Seg_data.size() << " segments at " << (frame_id + 1) << " sec " << std::endl;
        std::vector<std::size_t>
            label_id_list = Label::get_label_id(frame_id);

        Eigen::VectorXd eigen_x = d.col(0),
                        eigen_y = d.col(1);
        std::vector<double> x(eigen_x.data(), eigen_x.data() + eigen_x.size()), y(eigen_y.data(), eigen_y.data() + eigen_y.size());
        plt::clf();
        plt::plot(x, y, std::map<std::string, std::string>({{"marker", "."}, {"color", "blue"}, {"linestyle", ""}, {"markersize", "3"}}));

        for (std::size_t &leg_id : label_id_list) {
            std::cout << "  " << leg_id << "-th segment is a leg at " << (frame_id + 1) << " sec " << std::endl;
            Eigen::VectorXd seg_x = Seg_data[leg_id].col(0),
                            seg_y = Seg_data[leg_id].col(1);
            std::vector<double> x_(seg_x.data(), seg_x.data() + seg_x.size()), y_(seg_y.data(), seg_y.data() + seg_y.size());
            plt::plot(x_, y_, std::map<std::string, std::string>({{"marker", "o"}, {"markerfacecolor", "none"}, {"linestyle", ""}, {"markersize", "5"}}));
        }
        plt::title(std::to_string(frame_id + 1) + " sec.");
        plt::pause(0.1);
    }
}