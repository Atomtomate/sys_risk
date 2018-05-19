//
// Created by julian on 5/17/18.
//

#include "EigenTests.hpp"


TEST(eigenMapToStdVec, testEigen)
{
    Eigen::MatrixXd t = Eigen::MatrixXd::Random(3,3);
    std::vector<double> t2;
    t2.resize(3*3);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Map(&t2[0], t.rows(), t.cols()) = t;
    Eigen::MatrixXd t3 = Eigen::Map<Eigen::MatrixXd>(&t2[0], 3, 3);
    for(size_t i = 0; i < t2.size(); i ++)
    {
        EXPECT_EQ(t(i), t2[i]) << "Mapping from Eigen::MatrixXd to std::vector<double> did not work as expected";
        EXPECT_EQ(t3(i), t2[i]) << "Mapping from std::vector<double> to Eigen::MatrixXd did not work as expected";
        //std::cout << t(i) << "\t" << t2[i] << "\t" << t3(i) << "\n";
    }
}