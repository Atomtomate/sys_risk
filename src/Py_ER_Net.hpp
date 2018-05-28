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

class Py_ER_Net {
private:
    boost::mpi::environment env;
    boost::mpi::communicator world;
    bool isGenerator;
    boost::mpi::communicator local;
public:

    Py_ER_Net()
    {
        isGenerator = (world.size() > 1) ? (world.rank() > 0) : 1;
        local = world.split(isGenerator ? 0 : 1);
    }

    void run_valuation(const unsigned int N, const double p, const double val, const int which_to_set, const double T, const double r)
    {
        ER_Network nNN(local, world, isGenerator, N, p, val, which_to_set, T, r);
        nNN.test_ER_valuation();
    }

};


#endif //VALUATION_PY_ER_NET_HPP
