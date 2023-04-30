#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <vector>

#include "adaboost.hpp"
#include "feature.hpp"
#include "label.hpp"
#include "loaddata.hpp"
#include "utile.hpp"

int main() {
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> data = LoadData::LoadData("../xy_data.txt");
    Eigen::MatrixXd Train_X(0, 5), Test_X(0, 5);
    Eigen::VectorXd Train_Y(0), Test_Y(0);

    for (std::size_t frame_id = 0; frame_id < 60; ++frame_id) {
        const Eigen::MatrixXd &d = data[frame_id];
        std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Seg_data = Utile::Segment(d);
        std::vector<std::size_t> label_id_list = Label::get_label_id(frame_id);
        Eigen::VectorXd tmp = -Eigen::VectorXd::Ones(Seg_data.size());
        for (std::size_t &label_id : label_id_list) tmp(label_id) = 1;
        Train_Y = Utile::VectorConcat(Train_Y, tmp);

        for (const Eigen::MatrixXd &seg : Seg_data) {
            auto [N, sigma, W, circularity, radius] = Feature::feature_extraction(seg);
            Eigen::MatrixXd tmp(1, 5);
            tmp << N, sigma, W, circularity, radius;
            Utile::AppendOneRow(Train_X, tmp);
        }
    }

    for (std::size_t frame_id = 60; frame_id < 120; ++frame_id) {
        const Eigen::MatrixXd &d = data[frame_id];
        std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Seg_data = Utile::Segment(d);
        std::vector<std::size_t> label_id_list = Label::get_label_id(frame_id);
        Eigen::VectorXd tmp = -Eigen::VectorXd::Ones(Seg_data.size());
        for (std::size_t &label_id : label_id_list) tmp(label_id) = 1;
        Test_Y = Utile::VectorConcat(Test_Y, tmp);

        for (const Eigen::MatrixXd &seg : Seg_data) {
            auto [N, sigma, W, circularity, radius] = Feature::feature_extraction(seg);
            Eigen::MatrixXd tmp(1, 5);
            tmp << N, sigma, W, circularity, radius;
            Utile::AppendOneRow(Test_X, tmp);
        }
    }

    Adaboost classifier;
    std::ifstream ifs("classifier_save");
    boost::archive::text_iarchive ia(ifs);
    ia >> classifier;
    ifs.close();

    Eigen::VectorXd Pred_Y = classifier.predict(Train_X);
    double acc = 0;
    for (std::size_t idx = 0; idx < Pred_Y.size(); ++idx) {
        if (Pred_Y(idx) == Train_Y(idx)) acc++;
    }
    acc /= Pred_Y.size();
    Eigen::MatrixXd confusion = Utile::cal_confusion_matrix(Train_Y, Pred_Y);
    std::cout << confusion << std::endl;
    std::cout << "Accuracy :" << acc << std::endl;
    Pred_Y = classifier.predict(Test_X);
    acc = 0;
    for (std::size_t idx = 0; idx < Pred_Y.size(); ++idx) {
        if (Pred_Y(idx) == Test_Y(idx)) acc++;
    }
    acc /= Pred_Y.size();
    confusion = Utile::cal_confusion_matrix(Test_Y, Pred_Y);
    std::cout << confusion << std::endl;
    std::cout << "Accuracy :" << acc << std::endl;
}
