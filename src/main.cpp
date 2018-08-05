/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "main.hpp"
#include "Py_ER_Net.hpp"


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



int main(int argc, char* argv[])
{
    // Logging initialization
    START_EASYLOGGINGPP(argc, argv);
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

    // MPI initialization
#ifdef USE_MPI
    LOG(INFO) << "SETTINGS: using MPI";
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    bool isGenerator = (world.size() > 1) ? (world.rank() > 0) : 1;
    boost::mpi::communicator local = world.split(isGenerator ? 0 : 1);
    NetwSim nNN(local, world, true);
#else
    LOG(INFO) << "SETTINGS: not using MPI";
    NetwSim nNN;
#endif

#ifdef NDEBUG
    LOG(INFO) << "SETTINGS: Running release version";
#else
    el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);
    el::Loggers::addFlag(el::LoggingFlag::HierarchicalLogging);
    el::Loggers::setLoggingLevel(el::Level::Global);
    LOG(INFO) << "SETTINGS: Running debug version";
#endif
    //RInside R(argc, argv);
    //test_stan_math();
    //N2_network n2NN;
    //n2NN.test_N2_valuation();


    //Py_ER_Net pn;
    //LOG(INFO) << pn.add(4,5);
    //pn.run_valuation(2, 0.7, 0.5, 2, 1, 0);



    //TODO: eigen matrix dimension missmatch on large size?!?!
    Eigen::IOFormat CleanFmt(2, 0, " ", "\n", "[", "]");
    std::cout << "N: ";
    int N_;
    std::cin >> N_;
    std::cout << "conn: ";
    double conn_;
    std::cin >> conn_;
    std::cout << "row sum: ";
    double val;
    std::cin >> val;
    std::vector<double> plist {0.0,0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    for(auto p : plist) {
        LOG(INFO) << "Generating for p " << p;
        nNN.test_init_network(N_, p, val, 2, 1.0, 0.0, 1.0);
        Eigen::MatrixXd test = nNN.get_M();
        LOG(INFO) <<"\n" << test.rightCols(N_);
        LOG(INFO) << test.rightCols(N_).colwise().sum();
        LOG(INFO) << test.rightCols(N_).rowwise().sum();
    }
    exit(0);
    for (int N : {N_})
    { //, 8, 16, 32}) {
        nNN.test_init_network(N, conn_/static_cast<double>(N) , val, 2, 1.0, 0.0, 1.0);
        auto res = nNN.run_valuation(300, 300);//10000, 500);
        std::cout << "results: " << std::endl;
        for(const auto& el : res)
           std::cout << el.first << ": \n" << el.second.format(CleanFmt) << std::endl << " ===========" << std::endl;
    }
    //for (double p : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
    //    NetwSim nNN(local, world, true, 10, p, 0.95, 2, 1.0, 0.0);
    //   nNN.run_valuation();
    //}

    //@TODO: write test that expects equal sresults on equal seed
    return 0;
}
