/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#ifndef VALUATION_NETWORK_SIM_HPP
#define VALUATION_NETWORK_SIM_HPP

#define USE_ACTUAL_CONN 0

#include <type_traits>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <limits>
#include <random>
#include <vector>

#include "trng/chi_square_dist.hpp"

#include "Utils.hpp"

#ifdef USE_MPI

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#endif

#include "Config.hpp"
#include "StudentT.hpp"
#include "MVarNormal.hpp"
#include "Sampler.hpp"
#include "StatAcc.hpp"
#include "BlackScholesNetwork.hpp"
#include "RndGraphGen.hpp"


typedef typename std::conditional<USE_EIGEN_ACC, Eigen::MatrixXd, double>::type AccType;
typedef typename std::map<int, std::unordered_map<std::string, Eigen::MatrixXd>> ResultType;


struct SimulationParameters
{
    const long iterations, N_networks;
    NetworkType net_t;
};

constexpr int deg_of_freedom = 8;
//const std::string io_deg_str("In/Out degree distribution");

class NetwSim {
    friend class Py_ER_Net;
private:

#ifdef USE_MPI
    const boost::mpi::communicator local;
    const boost::mpi::communicator world;
    const bool isGenerator;
#endif

    trng::yarn2 gen_u;
    trng::uniform01_dist<> u_dist;
    Student_t_dist t_dist;
    Multivariate_Normal_Dist mvndist;
    //std::vector<double> dbg_weights;
    double last_weight;

    NetworkType net_t;
    int N;
    bool initialized;
    double T;              // maturity
    double r;              // interest
    double p;
    double val;
    double S0scalar;
    double sigmaScalar;
    double default_prob_scale;
    int setM;
    const double tmp[2][2] = {{1, 0},
                              {0, 1}};
    long Num_Samples;
    long Num_Networks;

    BlackScholesNetwork* bsn;
    Eigen::MatrixXd iSigma;
    Eigen::VectorXd sigma;
    Eigen::VectorXd Z;                 // Multivariate normal, used to generate lognormal assets
    Eigen::VectorXd var_h;
    Eigen::VectorXd S0;
    Eigen::VectorXd debt;
    Eigen::MatrixXd io_deg_dist;
    Eigen::MatrixXd avg_rc_sums;
    std::pair<double, double> avg_io_deg;

    std::map<int, std::unordered_map<std::string, Eigen::MatrixXd> > results;
    double connectivity;


    // last result, returned by observe
    void reset_network();

    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd* M);

    void init_2DFixed_BS(const double vs01, const double vs10, const double vr01, const double vr10)
    {
        connectivity = 1;
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(2, 4);
        Utils::fixed_2d(&M, vs01, vs10, vr01, vr10);
        io_deg_dist += Utils::in_out_degree(&M);
        avg_io_deg = Utils::avg_io_deg(&M);
        avg_rc_sums += Utils::avg_row_col_sums(&M);
        bsn->re_init(M, S0, debt, sigma);
    }

    template <typename F>
    void init_BS(F gen_function) {
        if (val < 0 || val >= 1) throw std::logic_error("Row sum is not in [0,1)");
        if (p < 0 || p > 1) throw std::logic_error("p is not a probability");
        connectivity = N * p;
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, 2 * N);
        gen_function(&M, gen_u, p, val, setM);
        //Utils::gen_fixed_degree(&M, gen_u, p, val, which_to_set);
        io_deg_dist += Utils::in_out_degree(&M);
        avg_io_deg = Utils::avg_io_deg(&M);
        avg_rc_sums += Utils::avg_row_col_sums(&M);
        bsn->re_init(M, S0, debt, sigma);
    }

//TODO: config struct
public:
    /*!
     * @brief               (re-)initializes network to given parameters
     * @param N             Size of network
     * @param p             Probability of cross holding
     * @param val           total value in/being held by other firms
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     * @TODO: config struct
     */
    void init_network(const int N_, const double p_, const double val_, const int which_to_set, const double T_,\
        const double r_, const double S0_, const double sigma_, const double default_prob_scale_, const NetworkType net_t_);

    void init_2D_network(BSParameters& bs_params, const double vs01, const double vs10, const double vr01, const double vr10);

    virtual ~NetwSim(){
        if(bsn != nullptr)
            delete bsn;
    }

    /*!
     * @brief               Constructs the Black Scholes Model using random cross holdings.
     * @param local         local MPI communicator (between producers/consumers only)
     * @param world         global MPI communicator
     * @param isGenerator   Flag for generator/consumer ranks
     */
#ifdef USE_MPI
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator):
            local(local), world(world), isGenerator(isGenerator), Z_dist(&tmp[0][0], &tmp[1][1]), chi_dist(deg_of_freedom), t_dist(deg_of_freedom)
#else
    NetwSim():
            Z_dist(&tmp[0][0], &tmp[1][1]), chi_dist(deg_of_freedom), t_dist(deg_of_freedom), initialized(false)
#endif
    {
        bsn = nullptr;
        iSigma = Eigen::MatrixXd::Zero(1,1);
        Z = Eigen::VectorXd::Zero(1,1);
        var_h = Eigen::VectorXd::Zero(1,1);
    }

    /*!
     * @brief               Constructs the Black Scholes Model using random cross holdings.
     * @param local         local MPI communicator (between producers/consumers only)
     * @param world         global MPI communicator
     * @param isGenerator   Flag for generator/consumer ranks
     * @param N             Size of network
     * @param p             Probability of connection between firms
     * @param val
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     * @param T             maturity
     * @param r             interest rate
     */
#ifdef USE_MPI
    NetwSim(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator,
               long N, double p, double val, int which_to_set, const double T, const double r, const double S0, const NetworkType net_t_) :
            local(local), world(world), isGenerator(isGenerator),
#else
    NetwSim(long N_, double p_, double val, int which_to_set, const double T_, const double r_, const double S0_, const double sigma_, const double default_scale_, const NetworkType net_t_) :
#endif
            val(val), T(T_), r(r_), S0scalar(S0_), sigmaScalar(sigma_), default_prob_scale(default_scale_)\
        , Z_dist(&tmp[0][0], &tmp[1][1]), chi_dist(deg_of_freedom), t_dist(deg_of_freedom), net_t(net_t_)
    {
        gen_u.seed();
        bsn = nullptr;
        init_network(N_, p_, val, which_to_set, T_, r_, S0_, sigma_, default_scale_, net_t_);
    }


    inline ResultType run_valuation(const SimulationParameters sim_params)
    {
        return run_valuation(sim_params.iterations, sim_params.N_networks);
    }

    /*!
     * @brief       Runs a series of example simulations
     * @param N_in  Size of network
     */
    ResultType run_valuation(const long N_Samples = 2000, const long N_networks = 100, const bool fix_degree = false);

    /*!
     * @brief   Draws a random number from a multivariate lognormal distribution
     * @return  Random sample from a multivariate lognormal distribution
     */
    const Eigen::MatrixXd draw_from_dist();

    double get_weight();

    /*!
     * @brief       Runs a single simulation of the Black Scholes model to find the fix point valuation.
     * @param St_in Initial asset value
     * @return      Valuation of firms at maturity T
     */
    auto run(const Eigen::Ref<const Eigen::VectorXd>& St_in)//Eigen::VectorXd St)
    {
        //Eigen::VectorXd St = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(St_in.data(), St_in.size());
        bsn->set_St(St_in);
        bsn->run_valuation(1000);
    }

    /*!
     * @brief   Compute \f$\Delta\f$ using the covariance matrix of the normal distribution
     * @return  \f$\Delta\f$
     */
    const Eigen::MatrixXd delta_v2();

    /*!
     * @brief   Computes the sum over all elements of the cross holdings matrix
     * @return  \f$\sum_{ij} M_{ij}\f$
     */
    std::vector<double> sumM() {
        std::vector<double> res{(bsn->get_M()).sum()};
        return res;
    }

    Eigen::MatrixXd get_M()
    {
        return bsn->get_M();
    }

    auto test_out()
    {
        auto v_o = bsn->get_valuation();
        auto s_o = bsn->get_solvent();
        std::cout << "output after sample: " << std::endl;
        LOG(INFO) << "Valuation: \n" << v_o;
        LOG(INFO) << "solvent: \n" << s_o;
        LOG(INFO) << "St: \n" << bsn->get_assets();
        LOG(INFO) << "debt: \n" << bsn->get_debt();
        LOG(INFO) << "M: \n" << bsn->get_M();
        std::cout << "------" << std::endl;
        Eigen::MatrixXd out = Eigen::MatrixXd::Constant(1,1,0);
        return out;
    }


private:

    trng::yarn2 gen_z;
    trng::yarn2 gen_chi;
    trng::chi_square_dist<double> chi_dist;
    trng::correlated_normal_dist<> Z_dist;


public:
    Eigen::MatrixXd get_io_deg_dist() const
    {
        return io_deg_dist;
    }

    Eigen::MatrixXd get_avg_row_col_sums() const
    {
        return avg_rc_sums;
    }

    void set_weight();
    //}

};


#endif //VALUATION_NETWORK_SIM_HPP
