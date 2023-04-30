#ifndef WEAKLEARNER_HPP
#define WEAKLEARNER_HPP

#include <Eigen/Eigen>
#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

/**
 * @brief The weak learner in Adaboost.
 */
class WeakLearner {
   private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version);
    double choose_value;  // w0 in the weight vector
    std::size_t choose_idx;
    double left_prob;
    double right_prob;

   public:
    WeakLearner() = default;
    void fit(const Eigen::ArrayXXd &train_X, const Eigen::ArrayXd &train_Y, const Eigen::ArrayXd &train_weight);
    Eigen::ArrayXd predict(const Eigen::ArrayXXd &section);
};

double gss(const std::function<double(double)> &f, double a, double b, const double &tol = 1e-10) {
    double R = 1.618033988749895;
    double sect_c = a + (b - a) / R;
    double sect_d = b - (b - a) / R;
    while (std::fabs(a - b) / 2 > tol) {
        if (f(sect_c) < f(sect_d)) {
            a = sect_d;
            sect_d = sect_c;
            sect_c = a + (b - a) / R;
        } else {
            b = sect_c;
            sect_c = sect_d;
            sect_d = b - (b - a) / R;
        }
    }
    return (a + b) / 2;
};

void WeakLearner::fit(const Eigen::ArrayXXd &train_X, const Eigen::ArrayXd &train_Y, const Eigen::ArrayXd &train_weight) {
    Eigen::ArrayXd train_Y_01 = (train_Y + 1) / 2;
    double score = std::numeric_limits<double>::infinity();
    std::size_t D = train_X.cols();
    for (std::size_t dim = 0; dim < D; ++dim) {
        Eigen::ArrayXd choose_data = train_X.col(dim);
        auto func = [&choose_data, &train_weight, &train_Y_01](double value) {
            Eigen::ArrayXd left = choose_data.unaryExpr([value](double x) { return static_cast<double>(x < value); });
            Eigen::ArrayXd right = (1 - left);
            Eigen::ArrayXd left_weight = left * train_weight, right_weight = right * train_weight;
            double left_weight_sum = left_weight.sum(), right_weight_sum = right_weight.sum();
            double left_prob = left_weight_sum == 0 ? 0 : (left_weight * train_Y_01).sum() / left_weight_sum;
            double right_prob = right_weight_sum == 0 ? 0 : (right_weight * train_Y_01).sum() / right_weight_sum;
            Eigen::ArrayXd pred_y = (left * left_prob + right * right_prob) * (1 - 2e-16) + 1e-16;
            return -((train_Y_01 * pred_y.log() + (1 - train_Y_01) * (1 - pred_y).log()) * train_weight).sum();
        };
        std::vector<double> value_candidate(10);
        double min_v = choose_data.minCoeff(), range_v = (choose_data.maxCoeff() - min_v) / 9;
        for (int i = 0; i < 10; ++i) {
            value_candidate[i] = gss(func, min_v + i * range_v, min_v + (i + 1) * range_v);
        }
        double tmp_value = *std::min_element(value_candidate.begin(), value_candidate.end(), [&func](const double &lhs, const double &rhs) { return func(lhs) < func(rhs); });  // gss(func, choose_data.minCoeff(), choose_data.maxCoeff());
        Eigen::ArrayXd left = choose_data.unaryExpr([&tmp_value](double x) { return static_cast<double>(x < tmp_value); });
        Eigen::ArrayXd right = (1 - left);
        Eigen::ArrayXd left_weight = left * train_weight, right_weight = right * train_weight;
        double left_weight_sum = left_weight.sum(), right_weight_sum = right_weight.sum();
        double tmp_left_prob = left_weight_sum == 0 ? 0 : (left_weight * train_Y_01).sum() / left_weight_sum;
        double tmp_right_prob = right_weight_sum == 0 ? 0 : (right_weight * train_Y_01).sum() / right_weight_sum;
        Eigen::ArrayXd pred_y = (left * tmp_left_prob + right * tmp_right_prob) * (1 - 2e-16) + 1e-16;
        double tmp_score = -((train_Y_01 * pred_y.log() + (1 - train_Y_01) * (1 - pred_y).log()) * train_weight).sum();
        if (tmp_score < score) {
            this->choose_idx = dim;
            this->choose_value = tmp_value;
            this->left_prob = tmp_left_prob;
            this->right_prob = tmp_right_prob;
            score = tmp_score;
        }
    }
}

Eigen::ArrayXd WeakLearner::predict(const Eigen::ArrayXXd &data) {
    Eigen::ArrayXd out = data.col(choose_idx).unaryExpr([this](double x) { return static_cast<double>(x < this->choose_value); });
    return (out * this->left_prob + (1 - out) * this->right_prob) * (1 - 2e-16) + 1e-16;
}

template <class Archive>
void WeakLearner::serialize(Archive &ar, const unsigned int version) {
    ar &this->choose_value &this->choose_idx &this->left_prob &this->right_prob;
}

#endif