#include "main.hpp"

using namespace Eigen;

/*void test_section6(void)
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
}*/

/*void test_eigenRCPP(RInside& R)
{
    std::string cmd = "set.seed(1); matrix(rnorm(9),3,3)";
    Eigen::Map<Eigen::MatrixXd> M = Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(R.parseEval(cmd));
    Eigen::MatrixXd N = M.transpose()*M;
    LOG(DEBUG) << M;
    LOG(DEBUG) << N;
}*/


Eigen::Matrix<stan::math::var, Dynamic, 1> test_stan_math_f(Eigen::Matrix<stan::math::var, Dynamic, 1> x_var)
{
    Eigen::Matrix<stan::math::var, Dynamic, 1> res = x_var; //(2*x_var.size());
    res << x_var(1), x_var(0);
    return res;
}

void test_stan_math() {
    Eigen::VectorXd x(2);
    x << 1.0,2.0;
    Eigen::Matrix<stan::math::var, Dynamic, 1> x_var(x.size());
    for (int i = 0; i < x.size(); ++i) 
        x_var(i) = x(i);
    Eigen::Matrix<stan::math::var, Dynamic, 1> f_x_var = test_stan_math_f(x_var);
    Eigen::VectorXd f_x(f_x_var.size());
    for (int i = 0; i < f_x.size(); ++i)
        f_x(i) = f_x_var(i).val();
    Eigen::MatrixXd J(f_x_var.size(), x_var.size());
    for (int i = 0; i < f_x_var.size(); ++i) {
        if (i > 0)
            stan::math::set_zero_all_adjoints();
        f_x_var(i).grad();
        for (int j = 0; j < x_var.size(); ++j)
            J(i,j) = x_var(j).adj();
    }
    LOG(INFO) << "x: \n" << x;
    LOG(INFO) << "f(x): \n" << f_x;
    LOG(INFO) << "J: \n" << J;
}


int main(int argc, char* argv[])
{
    // Logging initialization
    START_EASYLOGGINGPP(argc, argv);

    // MPI initialization
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    bool isGenerator = (world.size() > 1) ? (world.rank() > 0) : 1;
    boost::mpi::communicator local = world.split(isGenerator ? 0 : 1);

    //RInside R(argc, argv);
    //test_stan_math();
    //N2_network n2NN;
    //n2NN.test_N2_valuation();


    for (int N : {2, 4, 8, 16, 32}) {
        ER_Network nNN(local, world, true, N, 1.0, 0.95, 2);
        nNN.test_ER_valuation();
    }
    for (double p : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
        ER_Network nNN(local, world, true, 10, p, 0.95, 2);
        nNN.test_ER_valuation(10);
    }

    //@TODO: write test that expects equal sresults on equal seed
    return 0;
}
