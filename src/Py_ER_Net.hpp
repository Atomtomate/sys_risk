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


    void run_valuation(const unsigned int N, const double p, const double val_row, const double val_col, const unsigned int which_to_set, const double T, const double r, const double S0, const double sigma, const long iterations, const long N_networks, const double default_prob_scale, const int net_t)
    {
        NetworkType nt = NetworkType::ER;
        switch(net_t)
        {
            case 0: nt = NetworkType::ER; break;
            case 1: nt = NetworkType::Fixed2D; break;
            case 2: nt = NetworkType::STAR; break;
            case 3: nt = NetworkType::RING; break;
            case 4: nt = NetworkType::ER_SCALED; break;
            case 5: nt = NetworkType::UNIFORM; break;
            default:
                LOG(ERROR) << "Network typ not intialized";
                std::cout <<  "Network typ not intialized" << std::endl;
        }
        LOG(TRACE) << "Initializing network";
        LOG(INFO) << "init. p = " << p << " row sum = " << val_row << ", col sum" << val_col << ", r = " << r << " T = " << T << " it = "  << iterations;
        if(val_row != val_col)
            LOG(WARNING) << "ignoring column sum value! for not val_col == val_row is required";
        er_net.init_network(N, p, val_row, which_to_set, T, r, S0, sigma, default_prob_scale, nt);
        LOG(TRACE) << "Network initialized";
        er_net.run_valuation(iterations, N_networks);
    }

    void run_2DFixed_valuation(const double vs01, const double vs10, const double vr01, const double vr10,
            const double T, const double r, const double S0, const double sigma,
            const long iterations, const double default_prob_scale)
    {
        struct BSParameters bs_params = {T, r, sigma, S0, default_prob_scale};
        struct SimulationParameters sim_params = {iterations, 1, NetworkType::Fixed2D};
        er_net.init_2D_network(bs_params, vs01, vs10, vr01, vr10);
        er_net.run_valuation(sim_params);
    }



    Eigen::MatrixXd get_io_deg_dist() const
    {
        return er_net.get_io_deg_dist();
    }

    Eigen::MatrixXd get_avg_row_col_sums() const
    {
        return er_net.get_avg_row_col_sums();
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

    Eigen::MatrixXd get_Prop_k(int k, std::string s) const
    {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
        {
            auto el2 = el->second.find(s);
            if(el2 != el->second.end()) {
                return el2->second.transpose();//['#Samples'];
            } else {
                throw std::runtime_error("Number of Samples not available");
            }
        }
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_N_samples(int k) const
    {
        return get_Prop_k(k, count_str);
    }

    Eigen::MatrixXd get_M(int k) const
    {
        return get_Prop_k(k, M_str);
    }

    Eigen::MatrixXd get_M_var(int k) const {
        return get_Prop_k(k, "Variance " + M_str);
    }

    Eigen::MatrixXd get_rs(int k) const {
        return get_Prop_k(k, rs_str);
    }

    Eigen::MatrixXd get_rs_var(int k) const {
        return get_Prop_k(k, "Variance " + rs_str);
    }

    Eigen::MatrixXd get_solvent(int k) const {
        return get_Prop_k(k, solvent_str);
    }

    Eigen::MatrixXd get_solvent_var(int k) const {
        return get_Prop_k(k, "Variance " + solvent_str);
    }
    Eigen::MatrixXd get_delta_jac(int k) const {
        return get_Prop_k(k, delta1_str);
    }

    Eigen::MatrixXd get_delta_jac_var(int k) const {
        return get_Prop_k(k, "Variance " + delta1_str);
    }

    Eigen::MatrixXd get_assets(int k) const {
        return get_Prop_k(k, assets_str);
    }

    Eigen::MatrixXd get_assets_var(int k) const {
        return get_Prop_k(k, "Variance " + assets_str);
    }

    Eigen::MatrixXd get_valuation(int k) const {
        return get_Prop_k(k, val_str);
    }

    Eigen::MatrixXd get_valuation_var(int k) const {
        return get_Prop_k(k, "Variance " + val_str);
    }


    Eigen::MatrixXd get_vega(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Vega")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_vega_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Vega")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_theta(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Theta")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_theta_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Theta")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_rho(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Rho")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_rho_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Rho")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_pi(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Pi")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }

    Eigen::MatrixXd get_pi_var(int k) const {
        auto el = er_net.results.find(k);
        if(el != er_net.results.end())
            return  el->second.find("Variance Pi")->second.transpose();//['#Samples'];
        throw std::runtime_error("Tried to extract invalid <k>");
    }




};


#endif //VALUATION_PY_ER_NET_HPP
