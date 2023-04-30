#ifndef ADABOOST_HPP
#define ADABOOST_HPP

#include <Eigen/Eigen>
#include <boost/serialization/vector.hpp>
#include <iostream>
#include <vector>

#include "normalize.hpp"
#include "weaklearner.hpp"

class Adaboost {
   private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version);
    int M = 0;                   // the number of weak_learner
    std::vector<WeakLearner> T;  // the vector of weak_learner
    Normalizer::MinMaxScaler normalizer;

   public:
    Adaboost() = default;
    Adaboost(const int M)
        : M{M}, T(M) {}

    void fit(const Eigen::ArrayXXd &train_X, const Eigen::ArrayXd &train_Y);  // training Adaboost
    Eigen::ArrayXd predict(const Eigen::ArrayXXd &test_X);                    // make prediction
};

void Adaboost::fit(const Eigen::ArrayXXd &train_X, const Eigen::ArrayXd &train_Y) {
    this->normalizer.fit(train_X);
    Eigen::ArrayXXd train_X_ = this->normalizer.transform(train_X);
    Eigen::ArrayXd w = Eigen::ArrayXd::Ones(train_X_.rows());

    for (int i = 0; i < M; ++i) {
        std::cout << "Training " + std::to_string(i + 1) + "-th Weak Learner" << std::endl;
        w /= w.sum();
        this->T[i].fit(train_X_, train_Y, w);  // pred_Y is the label it predict, err is the error rate.
        Eigen::ArrayXd pred_Y = this->T[i].predict(train_X_);
        Eigen::ArrayXd Gm = (pred_Y / (1 - pred_Y)).log() / 2;
        w = w * (-train_Y * Gm).exp();
    }
}

Eigen::ArrayXd Adaboost::predict(const Eigen::ArrayXXd &data) {
    Eigen::ArrayXXd data_ = this->normalizer.transform(data);
    std::size_t R = data.rows();
    Eigen::ArrayXd C = Eigen::ArrayXd::Zero(R);
    for (std::size_t m = 0; m < M; ++m) {
        Eigen::ArrayXd pred_Y = this->T[m].predict(data_);
        C += (pred_Y / (1 - pred_Y)).log() / 2;
    }
    return 2 * C.unaryExpr([](double x) { return static_cast<double>(x > 0); }) - 1;
}

template <class Archive>
void Adaboost::serialize(Archive &ar, const unsigned int version) {
    ar &this->M &this->normalizer &this->T;
}
#endif