/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#ifndef VALUATION_ER_NETWORK_HPP
#define VALUATION_ER_NETWORK_HPP

#include <cstdlib>

#ifdef USE_MPI

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#endif

#include "Sampler.hpp"
#include "StatAcc.hpp"
#include "BlackScholesNetwork.hpp"


class ER_Network {
    friend class Py_ER_Net;
private:

#ifdef USE_MPI
    const boost::mpi::communicator local;
    const boost::mpi::communicator world;
    const bool isGenerator;
#endif

    unsigned int N;
    double T;              // maturity
    double r;              // interest
    double p;
    double val;
    unsigned int setM;
    const double tmp[2][2] = {{1, 0},
                              {0, 1}};
    BlackScholesNetwork* bsn;
    Eigen::MatrixXd itSigma;
    Eigen::VectorXd Z;                 // Multivariate normal, used to generate lognormal assets
    Eigen::VectorXd var_h;
    MCUtil::Sampler<Eigen::MatrixXd> S;

    // last result, returned by observers
    Eigen::VectorXd rs;

    void init_M_ER(double p, double val, unsigned int which_to_set, const Eigen::VectorXd& s0, const Eigen::VectorXd& debt);

public:
    /*!
     * @brief               (re-)initializes network to given parameters
     * @param N             Size of network
     * @param p             Probability of cross holding
     * @param val           Value of cross holding
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     */
    void init_network(const unsigned int N, const double p, const double val, const unsigned int which_to_set, const double T_new, const double r_new);
    void init_network(unsigned int N, double p, double val, unsigned int which_to_set);

    virtual ~ER_Network(){
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
            local(local), world(world), isGenerator(isGenerator), Z_dist(&tmp[0][0], &tmp[1][1])
#else
ER_Network():
            Z_dist(&tmp[0][0], &tmp[1][1])
#endif
    {
        bsn = nullptr;
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
     * @param val           Value of connection between firms
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     * @param T             maturity
     * @param r             interest rate
     */
#ifdef USE_MPI
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator,
               unsigned int N, double p, double val, unsigned int which_to_set, const double T, const double r) :
            local(local), world(world), isGenerator(isGenerator),
#else
    ER_Network(unsigned int N, double p, double val, unsigned int which_to_set, const double T, const double r) :
#endif
            T(T), r(r),
            Z_dist(&tmp[0][0], &tmp[1][1])
    {
        bsn = new BlackScholesNetwork(T, r);
        //@TODO: better Z_dist init
        //@TODO: assertions here, move expect to tests
        //EXPECT_GT(p, 0) << "p is not a probability";
        //EXPECT_LE(p, 1) << "p is not a probability";
        //EXPECT_GT(val, 0) << "val is not a probability";
        //EXPECT_LE(val, 1) << "val is not a probability";
        init_network(N, p, val, which_to_set);
        }



    inline void test_ER_valuation() { test_ER_valuation(N); };

    /*!
     * @brief       Runs a series of example simulations
     * @param N_in  Size of network
     */
    void test_ER_valuation(const unsigned int N_in, const unsigned int N_Samples = 10000);

    /*!
     * @brief   Draws a random number from a multivariate lognormal distribution
     * @return  Random sample from a multivariate lognormal distribution
     */
    Eigen::MatrixXd draw_from_dist();

    /*!
     * @brief       Runs a single simulation of the Black Scholes model to find the fix point valuation.
     * @param St_in Initial asset value
     * @return      Valuation of firms at maturity T
     */
    auto run(const Eigen::Ref<const Eigen::VectorXd>& St_in)//Eigen::VectorXd St)
    {
        //Eigen::VectorXd St = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(St_in.data(), St_in.size());
        bsn->set_St(St_in);
        rs = bsn->run_valuation(1000);
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

    auto test_out()
    {
        auto v_o = bsn->get_valuation();
        auto s_o = bsn->get_solvent();
        std::cout << "output after sample: " << std::endl;
        for(int i=0; i < v_o.size(); i++)
            std::cout << v_o[i] << "\t" << s_o[i] << std::endl;
        std::cout << "------" << std::endl;
        std::vector<double> out {0.};
        return out;
    }


private:

    trng::yarn2 gen_z;
    trng::correlated_normal_dist<> Z_dist;

};


#endif //VALUATION_ER_NETWORK_HPP
