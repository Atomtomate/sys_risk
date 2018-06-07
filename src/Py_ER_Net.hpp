/* Copyright (C) 5/28/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_PY_ER_NET_HPP
#define VALUATION_PY_ER_NET_HPP

#include "ER_Network.hpp"

class Py_ER_Net
{

private:
#ifdef USE_MPI
    boost::mpi::environment env;
    boost::mpi::communicator world;
    bool isGenerator;
    boost::mpi::communicator local;
#endif
    ER_Network er_net;

public:

#ifdef USE_MPI
    Py_ER_Net():
        er_net(local, world, isGenerator)
    {
        isGenerator = (world.size() > 1) ? (world.rank() > 0) : 1;
        local = world.split(isGenerator ? 0 : 1);
    }
#endif

    double add(int i, int j) { return i + j;}

    void run_valuation(const unsigned int N, const double p, const double val, const unsigned int which_to_set, const double T, const double r)
    {
        er_net.init_network(N, p, val, which_to_set, T, r);
        er_net.test_ER_valuation();
    }

    Eigen::MatrixXd get_M() const
    {
        return (er_net.bsn)->get_M();
    }


    Eigen::MatrixXd get_rs() const {
        return (er_net.bsn)->get_rs();
    }

    Eigen::MatrixXd get_solvent() const {
        return (er_net.bsn)->get_solvent();
    }

};


#endif //VALUATION_PY_ER_NET_HPP