/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#ifndef VALUATION_ER_NETWORK_HPP
#define VALUATION_ER_NETWORK_HPP

#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <string>
#include <unordered_map>

#include "Utils.hpp"

#ifdef USE_MPI

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#endif

#include "Sampler.hpp"
#include "StatAcc.hpp"
#include "BlackScholesNetwork.hpp"
#include "RndGraphGen.hpp"

struct Parameters
{
    long N;                 // M.rows()
    int set_s_d_both;       //
    double p;               // P(M_ij = 1)
    double val_row;             // sum_j M_ij
    double val_col;             // sum_j M_ij
    //@TODO: finish, add log-norm params


};

class ER_Network {
    friend class Py_ER_Net;
private:

#ifdef USE_MPI
    const boost::mpi::communicator local;
    const boost::mpi::communicator world;
    const bool isGenerator;
#endif
    trng::yarn2 gen_u;
    trng::uniform01_dist<> u_dist;

    long N;
    bool initialized;
    double T;              // maturity
    double r;              // interest
    double p;
    double val_row;
    double val_col;
    int setM;
    const double tmp[2][2] = {{1, 0},
                              {0, 1}};
    BlackScholesNetwork* bsn;
    Eigen::MatrixXd itSigma;
    Eigen::VectorXd Z;                 // Multivariate normal, used to generate lognormal assets
    Eigen::VectorXd var_h;
    MCUtil::Sampler<Eigen::MatrixXd>* S;
    Eigen::VectorXd S0;
    Eigen::VectorXd debt;
    Eigen::MatrixXd io_deg_dist;

    Eigen::MatrixXd count;
    Eigen::MatrixXd mean_delta_jac;
    Eigen::MatrixXd mean_delta_log;
    Eigen::MatrixXd mean_assets;
    Eigen::MatrixXd mean_rs;
    Eigen::MatrixXd mean_M;
    Eigen::MatrixXd mean_solvent;
    Eigen::MatrixXd mean_valuation;
    Eigen::MatrixXd mean_io_deg_dist;
    Eigen::MatrixXd var_delta_jac;
    Eigen::MatrixXd var_delta_log;
    Eigen::MatrixXd var_assets;
    Eigen::MatrixXd var_rs;
    Eigen::MatrixXd var_M;
    Eigen::MatrixXd var_solvent;
    Eigen::MatrixXd var_valuation;
    Eigen::MatrixXd var_io_deg_dist;
    double connectivity;


    // last result, returned by observe
    void test_init_network();

    void init_M_ER(const double p, const double val_row, const double val_col, const int which_to_set);

    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd* M);

public:
    /*!
     * @brief               (re-)initializes network to given parameters
     * @param N             Size of network
     * @param p             Probability of cross holding
     * @param val_row       total value in other firms
     * @param val_col       total value being held by others
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     */
    void test_init_network(const long N, const double p, const double val_row, const double val_col, const int which_to_set, const double T_new, const double r_new, const double default_prob_scale = 1.0);

    virtual ~ER_Network(){
        if(bsn != nullptr)
            delete bsn;
        if(S != nullptr)
            delete S;
    }

    /*!
     * @brief               Constructs the Black Scholes Model using random cross holdings.
     * @param local         local MPI communicator (between producers/consumers only)
     * @param world         global MPI communicator
     * @param isGenerator   Flag for generator/consumer ranks
     */
#ifdef USE_MPI
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator):
            local(local), world(world), isGenerator(isGenerator), Z_dist(&tmp[0][0], &tmp[1][1])
#else
    ER_Network():
            Z_dist(&tmp[0][0], &tmp[1][1]), initialized(false)
#endif
    {
        bsn = nullptr;
        S = new MCUtil::Sampler<Eigen::MatrixXd>();
        itSigma = Eigen::MatrixXd::Zero(1,1);
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
     * @param val_row
     * @param val_col
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     * @param T             maturity
     * @param r             interest rate
     */
#ifdef USE_MPI
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator,
               long N, double p, double val_row, double val_col, int which_to_set, const double T, const double r) :
            local(local), world(world), isGenerator(isGenerator),
#else
    ER_Network(long N_, double p_, double val_row, double val_col, int which_to_set, const double T_, const double r_) :
#endif
            Z_dist(&tmp[0][0], &tmp[1][1])
    {
        gen_u.seed();
        bsn = nullptr;
        S = new MCUtil::Sampler<Eigen::MatrixXd>();
        test_init_network(N_, p_, val_row, val_col, which_to_set, T_, r_);
    }



    /*!
     * @brief       Runs a series of example simulations
     * @param N_in  Size of network
     */
    std::unordered_map<std::string, Eigen::MatrixXd> test_ER_valuation(const long N_Samples = 2000, const long N_networks = 100);

    /*!
     * @brief   Draws a random number from a multivariate lognormal distribution
     * @return  Random sample from a multivariate lognormal distribution
     */
    const Eigen::MatrixXd draw_from_dist();

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
        std::cout << "------" << std::endl;
        Eigen::MatrixXd out = Eigen::MatrixXd::Constant(1,1,0);
        return out;
    }


private:

    trng::yarn2 gen_z;
    trng::correlated_normal_dist<> Z_dist;

};


#endif //VALUATION_ER_NETWORK_HPP
