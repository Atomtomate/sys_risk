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
    const double T = 1.0;              // maturity
    const double r = 0.0;              // interest
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


public:
    void init_network(unsigned int N, double p, double val, char which_to_set);

    //@TODO: better Z_dist init
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator,
               unsigned int N, double p, double val, char which_to_set) :
            local(local), world(world), isGenerator(isGenerator), bsn(Eigen::MatrixXd::Zero(N, 2 * N), T, r),
            Z_dist(&tmp[0][0], &tmp[1][1]) {
        init_network(N, p, val, which_to_set);
    }

    inline void test_ER_valuation() { test_ER_valuation(N); };

    void test_ER_valuation(const unsigned int N_in);

    std::vector<double> draw_from_dist();

    auto run(std::vector<double> St_in)//Eigen::VectorXd St)
    {
        Eigen::VectorXd St = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(St_in.data(), St_in.size());
        bsn.set_St(St);
        rs = bsn.run_valuation(1000);
    }


    std::vector<double> delta_v2();

    std::vector<double> sumM() {
        std::vector<double> res{bsn.get_M().sum()};
        return res;
    }


private:

    trng::yarn2 gen_z;
    trng::correlated_normal_dist<> Z_dist;

};


#endif //VALUATION_ER_NETWORK_HPP
