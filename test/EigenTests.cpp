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

TEST(eigenBoostSerialize, testEigen)
{
    Eigen::MatrixXd A(4,4);
    A << 1,2,3,4,5,6,7,8,9,10,11, 12,13,14,15,16;
    LOG(INFO) << "writing: \n" << A;
    LOG(INFO) << "serializing";

    auto res = MCUtil::write_to_binary(A, "test_out");
    Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(4,4);
    LOG(INFO) << "Writing to binary file returned: " << res;
    res = MCUtil::read_from_binary(A2, "test_out");
    LOG(INFO) << "reading from binary file returned: " << res;
    LOG(INFO) << "Read: \n" << A2;
    //ASSERT_TRUE(A.isApprox(A2));

    // test text
    A2 = Eigen::MatrixXd::Zero(4,4);
    res = MCUtil::write_to_text(A, "test_out_txt.txt");
    LOG(INFO) << "Writing to text file returned: " << res;
    res = MCUtil::read_from_text(A2, "test_out_txt.txt");
    LOG(INFO) << "reading from text file returned: " << res;
    LOG(INFO) << "Read: \n" << A2;
    //ASSERT_TRUE(A.isApprox(A2));


    // test xml
    /*A2 = Eigen::MatrixXd::Zero(4,4);
    res = MCUtil::write_to_xml(A, "test_out.xml", "A");
    LOG(INFO) << "Writing to xml file returned: " << res;
    res = MCUtil::read_from_xml(A2, "test_out.xml", "A");

    LOG(INFO) << "reading from xml file returned: " << res;
    LOG(INFO) << "Read: \n" << A2;
    //ASSERT_TRUE(A.isApprox(A2));*/
}