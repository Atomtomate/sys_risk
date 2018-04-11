#include "main.hpp"

using namespace Eigen;

void test_section6(void)
{
    MatrixXd vij = MatrixXd::Identity(4,4);
    MatrixXd zij(4,8);
    zij << .0,.2,.3,.1,.0,.0,.1,.0,  .2,.0,.2,.1,.0,.0,.0,.1,\
                 .1,.1,.0,.3,.1,.0,.0,.1,   .1,.1,.1,.0,.0,.1,.0,.0;
    VectorXd B(4);
    B << .8, .8, .8, .8;
    unsigned int L = 20;
    unsigned int max_it = 1000;
    for(int L = 10000; L < 1000000; L*=4)
    {
        LOG(INFO) << "running MC";
        auto res = run_valuation(vij, zij, B, 4, 20, L);
        LOG(INFO) << "V_final: \n" << res;
    }
}

void test_eigenRCPP(RInside& R)
{
    std::string cmd = "set.seed(1); matrix(rnorm(9),3,3)";
    Eigen::Map<Eigen::MatrixXd> M = Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(R.parseEval(cmd));
    Eigen::MaxtrixXd N = M.transpose()*M;
    LOG(DEBUG) << M;
    LOG(DEBUG) << N;
}

int main(int argc, char* argv[])
{
    START_EASYLOGGINGPP(argc, argv);
    RInside R(argc, argv);
    
    test_eigenRCPP(R);
    //figure6();
    return 0;
}
