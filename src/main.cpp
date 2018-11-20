/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "main.hpp"
#include "Py_ER_Net.hpp"
#include "MVarNormal.hpp"
#include "StudentT.hpp"


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

    boost::property_tree::ptree pt;
    boost::property_tree::info_parser::read_info("config.info", pt);
    int N_ = pt.get("ModelParameters.size", 10);
    int netType_i = pt.get("ModelParameters.NetworkType", 0); // 0 - ER, 1 - Fixed2d, 2 - star, 3 - ring
    int setM =  pt.get("ModelParameters.BothDebtEq", 2);
    double r_ = pt.get("ModelParameters.r", 0.0);
    double T_ = pt.get("ModelParameters.T", 1.0);
    double S0_ = pt.get("ModelParameters.S0", 1.0);
    double conn_ = pt.get("ModelParameters.connectivity", 0.5);
    double val = pt.get("ModelParameters.colSums", 0.5);
    double ds = pt.get("ModelParameters.defaultScale", 1.0);
    double sigma_ = pt.get("ModelParameters.sigma", 0.1);
    int runAvg = pt.get("ProgramOptions.averageOverNetwork", 0);
    int N_nets = pt.get("ProgramOptions.NumberOfNets", 500);
    int N_MC = pt.get("ProgramOptions.NumberOfMCSamples", 200);
    std::cout << "running with: ";
    std::cout << "Network Type: " << netType_i << ", ";
    std::cout << "Both/Debt/Equity: " << setM << ", ";
    std::cout << "N: " << N_ << ", ";
    std::cout << "conn: " << conn_ << ", ";
    std::cout << "r: " << r_ << ", ";
    std::cout << "T: " << T_ << ", ";
    std::cout << "val: " << val << ", ";
    std::cout << "sigma: " << sigma_ << ", ";
    std::cout << "default scale: " << ds << std::endl;

    NetworkType net_t = NetworkType::ER;;
    switch(netType_i)
    {
        case 0:
            net_t = NetworkType::ER; break;
        case 1:
            net_t = NetworkType::Fixed2D; break;
        case 2:
            net_t = NetworkType::STAR; break;
        case 3:
            net_t = NetworkType::RING; break;
        case 4:
            net_t = NetworkType::ER_SCALED; break;
        case 5:
            net_t = NetworkType::UNIFORM; break;
        default:
            LOG(ERROR) << "Uninitialized network type";
    }
    /*std::vector<double> plist { 0.7, 0.9, 1.0}; //0.0,0.1,0.2, 0.3, 0.4,0.5, 0.6,0.8,
    for(auto p : plist) {
        LOG(INFO) << "Generating for p " << p;
        nNN.init_network(5, 1.0, 0.7, 2, 1.0, 0.0, 1.0, 0.15, 1.0);

        Eigen::MatrixXd M = Eigen::MatrixXd::Random(3,6);
        Eigen::MatrixXd test = M;
        LOG(WARNING) << test;
        LOG(INFO) <<"\n" << test.rightCols(N_);
        LOG(INFO) << test.rightCols(N_).colwise().sum();
        LOG(INFO) << test.rightCols(N_).rowwise().sum();
    }
    exit(0);
    */
    BSParameters bsp = {T_, r_, sigma_, S0_, ds};


    for (int N : {N_})
    { //, 8, 16, 32}) {
        if(netType_i != 1)
            nNN.init_network(N, conn_/static_cast<double>(N) , val, setM, T_, r_, S0_, sigma_, ds, net_t);
        else
            nNN.init_2D_network(bsp, 0.9, 0.9, 0.9, 0.9);
        auto res = nNN.run_valuation(N_MC, N_nets);//10000, 500);
        std::cout << "results: " << std::endl;
        for(auto res_el: res ) {
            std::cout << "-=-=-=-=-=-=-=-=-=-=-=- <k> = " << res_el.first/static_cast<double>(COARSE_CONN) << " -=-=-=-=-=-=-=-=-=-=-=-" << std::endl;
            for (const auto &el : res_el.second)
            {
#if USE_EIGEN_ACC
                if(!runAvg || USE_EIGEN_ACC)
                {
                    std::cout << el.first << ": " << std::endl << el.second.format(CleanFmt);
                    std::cout  << std::endl << " ===========" << std::endl;
                }
                else
                {
                    auto res = el.second;
                    if(res.cols() > N_)
                    {
                        std::cout << el.first;
                        double norm = N*res.rows();
                        std::cout << " equity: " << res.leftCols(N).sum()/norm << std::endl;
                        std::cout << " debt: " << res.rightCols(N).sum()/norm << std::endl;

                    }
                    if(res.rows() > N_)
                    {
                        std::cout << el.first;
                        double norm = N*res.cols();
                        std::cout << " equity: " << res.topRows(N).sum()/norm << std::endl;
                        std::cout << " debt: " << res.bottomRows(N).sum()/norm << std::endl;
                    }
                    else
                    {
                        double norm = res.size();
                        std::cout << el.first;
                        std::cout << ": " << res.sum()/norm << std::endl;
                    }
                }
#else
                std::cout << ": " << std::endl << el.second;
                std::cout  << std::endl << " ===========" << std::endl;

#endif
            }
        }
        std::cout << " Avg IO degree distribution: " << std::endl;
        std::cout << nNN.get_io_deg_dist() << std::endl;

        std::cout << " =========================== " << std::endl;
        std::cout << " Avg IO weight distribution: " << std::endl;
        std::cout << nNN.get_avg_row_col_sums() << std::endl;
        std::cout << " =========================== " << std::endl;



    }
    //for (double p : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
    //    NetwSim nNN(local, world, true, 10, p, 0.95, 2, 1.0, 0.0);
    //   nNN.run_valuation();
    //}

    //@TODO: write test that expects equal sresults on equal seed
    return 0;
}
