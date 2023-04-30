#ifndef NORMALIZER_HPP
#define NORMALIZER_HPP
#include <Eigen/Eigen>
#include <boost/serialization/vector.hpp>
namespace Normalizer {

class StandardScaler {
   private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar &this->data_mean &this->data_std;
    }
    std::vector<double> data_mean;  // the minumum num of each column, my feature matrix have 5 column, thus the size of data_min is 5
    std::vector<double> data_std;   // the max-min num of each column, my feature matrix have 5 column, thus the size of data_mm is 5

   public:
    StandardScaler() = default;
    void fit(const Eigen::ArrayXXd &data) {
        const std::size_t COLS = data.cols();
        this->data_mean = std::vector<double>(COLS, 0);
        this->data_std = std::vector<double>(COLS, 0);

        for (int i = 0; i < COLS; ++i) {
            Eigen::ArrayXd choose_data = data.col(i);
            this->data_mean[i] = choose_data.mean();
            this->data_std[i] = std::sqrt((choose_data - this->data_mean[i]).square().sum() / choose_data.size());  // max - min
            if (this->data_std[i] == 0) this->data_std[i] = 1;
        }
    }
    Eigen::ArrayXXd transform(const Eigen::ArrayXXd &data) {
        Eigen::ArrayXXd tf_matrix(data.rows(), data.cols());

        const int COLS = data.cols();
        for (int i = 0; i < COLS; ++i)
            tf_matrix.col(i) = (data.col(i) - this->data_mean[i]) / this->data_std[i];

        return tf_matrix;
    }
};

class MinMaxScaler {
   private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar &this->data_min &this->data_range;
    }
    std::vector<double> data_min;    // the minumum num of each column, my feature matrix have 5 column, thus the size of data_min is 5
    std::vector<double> data_range;  // the max-min num of each column, my feature matrix have 5 column, thus the size of data_mm is 5

   public:
    MinMaxScaler() = default;
    void fit(const Eigen::ArrayXXd &data) {
        const std::size_t COLS = data.cols();
        this->data_min = std::vector<double>(COLS, 0);
        this->data_range = std::vector<double>(COLS, 0);

        for (int i = 0; i < COLS; ++i) {
            Eigen::ArrayXd choose_data = data.col(i);
            this->data_min[i] = choose_data.minCoeff();
            this->data_range[i] = choose_data.maxCoeff() - this->data_min[i];  // max - min
            if (this->data_range[i] == 0) this->data_range[i] = 1;
        }
    }
    Eigen::ArrayXXd transform(const Eigen::ArrayXXd &data) {
        Eigen::ArrayXXd tf_matrix(data.rows(), data.cols());

        const int COLS = data.cols();
        for (int i = 0; i < COLS; ++i)
            tf_matrix.col(i) = (data.col(i) - this->data_min[i]) / this->data_range[i];

        return tf_matrix;
    }
};

}  // namespace Normalize
#endif