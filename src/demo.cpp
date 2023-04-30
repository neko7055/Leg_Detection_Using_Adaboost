#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <boost/archive/text_iarchive.hpp>
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
    Adaboost classifier;
    std::ifstream ifs("classifier_save");
    boost::archive::text_iarchive ia(ifs);
    ia >> classifier;
    ifs.close();
    for (std::size_t frame_id = 0; frame_id < data.size(); frame_id++) {
        const Eigen::MatrixXd &d = data[frame_id];
        std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Seg_data = Utile::Segment(d);
        Eigen::MatrixXd X(0, 5);
        for (const Eigen::MatrixXd &seg : Seg_data) {
            auto [N, sigma, W, circularity, radius] = Feature::feature_extraction(seg);
            Eigen::MatrixXd tmp(1, 5);
            tmp << N, sigma, W, circularity, radius;
            Utile::AppendOneRow(X, tmp);
        }
        Eigen::VectorXd Pred_Y = classifier.predict(X);

        Eigen::VectorXd eigen_x = d.col(0),
                        eigen_y = d.col(1);
        std::vector<double> x(eigen_x.data(), eigen_x.data() + eigen_x.size()), y(eigen_y.data(), eigen_y.data() + eigen_y.size());
        plt::clf();
        plt::plot(x, y, std::map<std::string, std::string>({{"marker", "."}, {"color", "blue"}, {"linestyle", ""}, {"markersize", "3"}}));
        for (std::size_t idx = 0; idx < Seg_data.size(); ++idx) {
            if (Pred_Y(idx) == 1) {
                Eigen::VectorXd seg_x = Seg_data[idx].col(0),
                                seg_y = Seg_data[idx].col(1);
                std::vector<double> x_(seg_x.data(), seg_x.data() + seg_x.size()), y_(seg_y.data(), seg_y.data() + seg_y.size());
                plt::plot(x_, y_, std::map<std::string, std::string>({{"marker", "o"}, {"markerfacecolor", "none"}, {"linestyle", ""}, {"markersize", "5"}}));
            }
        }
        plt::title(std::to_string(frame_id + 1) + " sec.");
        plt::pause(0.1);
    }
}