#ifndef UTILE_HPP
#define UTILE_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <array>
#include <vector>

namespace Utile {
inline double square(double x) { return x * x; }
inline double distance(const std::array<double, 2> &vec1, const std::array<double, 2> &vec2) {
    return std::sqrt(square(vec1[0] - vec2[0]) + square(vec1[1] - vec2[1]));
}

Eigen::VectorXd VectorConcat(const Eigen::VectorXd &A, const Eigen::VectorXd &B) {
    Eigen::VectorXd output(A.size() + B.size());
    output << A, B;
    return output;
}

inline void AppendOneRow(Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
    std::size_t len = A.rows(), dim = A.cols();
    A.conservativeResize(len + 1, A.cols());
    for (std::size_t d = 0; d < dim; ++d) A(len, d) = B(0, d);
}

inline void AppendTwoMatToB(const Eigen::MatrixXd &A, Eigen::MatrixXd &B) {
    Eigen::MatrixXd D(A.rows() + B.rows(), A.cols());
    D << A, B;
    B = D;
}

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Segment(const Eigen::MatrixXd &origin_data, double threshold = 0.1) {
    std::vector<std::array<double, 2>> data;
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Returndata;
    for (std::size_t i = 0; i < origin_data.rows(); i++) {
        if (((origin_data(i, 0) != 0) || (origin_data(i, 1) != 0)) && (std::isfinite(origin_data(i, 0)) && std::isfinite(origin_data(i, 1)))) {
            // std::array<double, 2> push_data = { origin_data[i][0], origin_data[i][1] };
            data.push_back({origin_data(i, 0), origin_data(i, 1)});
        }
    }

    std::size_t validsize = data.size(), ReturnEndIDX = 0;
    bool one_is_end = distance(data[0], data[validsize - 1]) < threshold;
    Eigen::MatrixXd tmp(1, 2);
    tmp << data[0][0], data[0][1];

    Returndata.emplace_back(tmp);

    for (int i = 1; i < validsize; i++) {
        tmp << data[i][0], data[i][1];
        if (distance(data[i - 1], data[i]) < threshold)
            AppendOneRow(Returndata[ReturnEndIDX], tmp);

        else {
            ReturnEndIDX++;
            Returndata.emplace_back(tmp);
        }
    }

    if (one_is_end) {
        AppendTwoMatToB(Returndata[ReturnEndIDX], Returndata[0]);
        Returndata.pop_back();
    }
    return Returndata;
}

Eigen::MatrixXd cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y) {
    std::size_t R = y.size();
    std::size_t TP{}, FP{}, FN{}, TN{};

    for (std::size_t i = 0; i < R; ++i) {
        if (y(i) == -1 && pred_Y(i) == -1)
            ++TN;
        else if (y(i) == -1 && pred_Y(i) == 1)
            ++FP;
        else if (y(i) == 1 && pred_Y(i) == 1)
            ++TP;
        else if (y(i) == 1 && pred_Y(i) == -1)
            ++FN;
    }

    Eigen::MatrixXd confusion(2, 2);
    confusion << TP, FP, FN, TN;
    return confusion;
}

}  // namespace Utile

#endif