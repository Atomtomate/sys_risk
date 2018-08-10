/* Copyright (C) 5/28/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_PY_ER_NET_HPP
#define VALUATION_PY_ER_NET_HPP

#include "NetwSim.hpp"

class Py_ER_Net
{

private:
#ifdef USE_MPI
    boost::mpi::environment env;
    boost::mpi::communicator world;
    bool isGenerator;
    boost::mpi::communicator local;
#endif
    NetwSim er_net;

public:

#ifdef USE_MPI
    Py_ER_Net():
        er_net(local, world, isGenerator)
    {
        isGenerator = (world.size() > 1) ? (world.rank() > 0) : 1;
        local = world.split(isGenerator ? 0 : 1);
    }
#endif


    void run_valuation(const unsigned int N, const double p, const double val_row, const double val_col, const unsigned int which_to_set, const double T, const double r, const long iterations, const long N_networks, const double default_prob_scale)
    {
        LOG(TRACE) << "Initializing network";
        LOG(INFO) << "init. p = " << p << " row sum = " << val_row << ", col sum" << val_col << ", r = " << r << " T = " << T << " it = "  << iterations;
        if(val_row != val_col)
            LOG(WARNING) << "ignoring column sum value! for not val_col == val_row is required";
        er_net.test_init_network(N, p, val_row, which_to_set, T, r, default_prob_scale);
        LOG(TRACE) << "Network initialized";
        er_net.run_valuation(iterations, N_networks);
    }

    Eigen::MatrixXi get_k_vals() const
    {
        if(er_net.results.size() == 0)
            throw std::runtime_error("No results available!");
        Eigen::MatrixXi res(1, er_net.results.size());
        auto it = er_net.results.begin();
        int i = 0;
        while(it != er_net.results.end())
        {
            res(0,i) = (int)(it->first);
            it++;
            i++;
        }
        return res;
    }

    Eigen::MatrixXd get_N_samples(int k) const
    {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("#Samples")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_M(int k) const
    {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("M")->second;//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_M_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance M")->second;//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_rs(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("RS")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_rs_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance RS")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_solvent(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Solvent")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_solvent_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Solvent")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }
    Eigen::MatrixXd get_delta_jac(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Delta using Jacobians")->second;//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_delta_jac_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Delta using Jacobians")->second;//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_assets(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Assets")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_assets_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Assets")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_valuation(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Valuation")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_valuation_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Valuation")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }


};


#endif //VALUATION_PY_ER_NET_HPP
