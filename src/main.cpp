#include "main.hpp"

using namespace Eigen;

int main(int argc, char* argv[])
{
    START_EASYLOGGINGPP(argc, argv);
    MatrixXd vij = MatrixXd::Identity(4,4);
    MatrixXd zij(4,8);
    zij << .0,.2,.3,.1,.0,.0,.1,.0,  .2,.0,.2,.1,.0,.0,.0,.1,\
                 .1,.1,.0,.3,.1,.0,.0,.1,   .1,.1,.1,.0,.0,.1,.0,.0;
    VectorXd B(4);
    B << .8, .8, .8, .8;
    unsigned int L = 20;
    unsigned int max_it = 1000;
    for(int L = 100; L < 100000; L*=2)
    {
        LOG(INFO) << "running MC";
        auto res = run_valuation(vij, zij, B, 4, 20, L);
        LOG(INFO) << res;
    }
    return 0;
}
