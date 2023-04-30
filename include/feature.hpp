#ifndef FEATURE_HPP
#define FEATURE_HPP
#include <Eigen/Dense>
#include <tuple>
namespace Feature {

std::size_t num_of_point(const Eigen::MatrixXd &Seg) {
    return Seg.rows();
}

std::tuple<double, double> circle_fit(const Eigen::MatrixXd &data) {
    int n = num_of_point(data);
    if (n < 3) return {0.0, 0.0};
    Eigen::MatrixXd A(data.rows(), data.cols() + 1);
    Eigen::MatrixXd b(data.rows(), 1);
    b << (-1 * data.array().square().matrix()) * Eigen::MatrixXd::Ones(data.cols(), 1);
    A << -2 * data, Eigen::MatrixXd::Ones(data.rows(), 1);
    Eigen::MatrixXd solution = Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>(A).solve(b);
    double xc = solution(0), yc = solution(1);
    double radius = std::sqrt(std::pow(xc, 2) + std::pow(yc, 2) - solution(2));
    double circularity = ((radius - ((xc - data.col(0).array()).square() + (yc - data.col(1).array()).square()).sqrt()).square()).sum();
    return {circularity, radius};
}

double std(const Eigen::MatrixXd &Seg) {
    int n = num_of_point(Seg);
    if (n < 2) return 0.0;

    Eigen::Vector2d m = Seg.colwise().mean();  // the means vector of data
    const auto &x = Seg.col(0);
    const auto &y = Seg.col(1);

    return std::sqrt(((x.array() - m(0)).square().sum() + (y.array() - m(1)).square().mean()));
}

double width(const Eigen::MatrixXd &Seg) {
    int n = num_of_point(Seg);
    if (n < 2) return 0.0;
    return std::sqrt(std::pow(Seg(0, 0) - Seg(Seg.rows() - 1, 0), 2) + std::pow(Seg(0, 1) - Seg(Seg.rows() - 1, 1), 2));
}

std::tuple<double, double, double, double, double> feature_extraction(const Eigen::MatrixXd &Seg) {
    auto [circularity, radius] = circle_fit(Seg);
    double N = num_of_point(Seg), sigma = std(Seg), W = width(Seg);
    return {N, sigma, W, circularity, radius};
}

}  // namespace Feature
#endif