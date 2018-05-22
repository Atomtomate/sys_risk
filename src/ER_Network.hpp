//
// Created by julian on 5/21/18.
//
#ifndef VALUATION_ER_NETWORK_HPP
#define VALUATION_ER_NETWORK_HPP

#include <cstdlib>


#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "Sampler.hpp"
#include "StatAcc.hpp"
#include "BlackScholesNetwork.hpp"


class ER_Network {
private:
    const boost::mpi::communicator local;
    const boost::mpi::communicator world;
    const bool isGenerator;
    unsigned int N;
    const double T;              // maturity
    const double r;              // interest
    double p;
    double val;
    char setM;
    const double tmp[2][2] = {{1, 0},
                              {0, 1}};
    BlackScholesNetwork bsn;
    Eigen::MatrixXd itSigma;
    Eigen::VectorXd Z;                 // Multivariate normal, used to generate lognormal assets
    Eigen::VectorXd var_h;

    // last result, returned by observers
    std::vector<double> rs;

    void set_M_ER(double p, double val, char which_to_set);

public:
    /*!
     * @brief               (re-)initializes network to given parameters
     * @param N             Size of network
     * @param p             Probability of cross holding
     * @param val           Value of cross holding
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     */
    void init_network(unsigned int N, double p, double val, char which_to_set);

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
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator,
               unsigned int N, double p, double val, char which_to_set, const double T, const double r) :
            local(local), world(world), isGenerator(isGenerator),
            T(T), r(r), bsn(Eigen::MatrixXd::Zero(N, 2 * N), T, r),
            Z_dist(&tmp[0][0], &tmp[1][1])
    {

            //@TODO: better Z_dist init
        EXPECT_GT(p, 0) << "p is not a probability";
        EXPECT_LE(p, 1) << "p is not a probability";
        EXPECT_GT(val, 0) << "val is not a probability";
        EXPECT_LE(val, 1) << "val is not a probability";
        init_network(N, p, val, which_to_set);
        }



    inline void test_ER_valuation() { test_ER_valuation(N); };

    /*!
     * @brief       Runs a series of example simulations
     * @param N_in  Size of network
     */
    void test_ER_valuation(const unsigned int N_in);

    /*!
     * @brief   Draws a random number from a multivariate lognormal distribution
     * @return  Random sample from a multivariate lognormal distribution
     */
    std::vector<double> draw_from_dist();

    /*!
     * @brief       Runs a single simulation of the Black Scholes model to find the fix point valuation.
     * @param St_in Initial asset value
     * @return      Valuation of firms at maturity T
     */
    auto run(std::vector<double> St_in)//Eigen::VectorXd St)
    {
        Eigen::VectorXd St = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(St_in.data(), St_in.size());
        bsn.set_St(St);
        rs = bsn.run_valuation(1000);
    }

    /*!
     * @brief   Compute \f$\Delta\f$ using the covariance matrix of the normal distribution
     * @return  \f$\Delta\f$
     */
    std::vector<double> delta_v2();

    /*!
     * @brief   Computes the sum over all elements of the cross holdings matrix
     * @return  \f$\sum_{ij} M_{ij}\f$
     */
    std::vector<double> sumM() {
        std::vector<double> res{bsn.get_M().sum()};
        return res;
    }


private:

    trng::yarn2 gen_z;
    trng::correlated_normal_dist<> Z_dist;

};


#endif //VALUATION_ER_NETWORK_HPP
