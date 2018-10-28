/* Copyright (C) 10/25/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_RESULTS_BS_HPP
#define VALUATION_RESULTS_BS_HPP

#include <type_traits>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <limits>
#include <random>
#include <vector>

#include "Config.hpp"
#include "Sampler.hpp"
#include "StatAcc.hpp"
#include "BlackScholesNetwork.hpp"

// TODO: use CRTP to make static interface

class Results_BS
{

private:
#if USE_EIGEN_ACC
    using AccT = MCUtil::StatAccEigen<double, 0>;
#else
    using AccT = MCUtil::StatAcc<double, 0>;
#endif
    std::unordered_map<std::string, AccT*> accs;
    int N;
    BlackScholesNetwork* bsn;
    int samples_count;

public:
    Results_BS &operator=(const Results_BS &) = delete;

    Results_BS(const Results_BS &) = delete;

    virtual ~Results_BS()
    {
        for(auto it = accs.begin(); it != accs.end(); it++)
        {
            delete (it->second);
        }
        accs.clear();
    }

    Results_BS(const int N_, BlackScholesNetwork* bsn_):
        N(N_), bsn(bsn_), samples_count(0)
    {
        if(bsn == nullptr) LOG(ERROR) << "bsn variable not initialized before registering observers!";
#if SINGLE_GREEKS
        accs.insert( std::make_pair(greeks_str, new AccT(4,2)) );
#else
        accs.insert( std::make_pair(delta1_str, new AccT(2*N,N)) );
        accs.insert( std::make_pair(rho_str, new AccT(2*N,1)) );
        accs.insert( std::make_pair(theta_str, new AccT(2*N,1)) );
        accs.insert( std::make_pair(vega_str, new AccT(2*N,N)) );
#endif
        accs.insert( std::make_pair(assets_str, new AccT(N,1)) );
        accs.insert( std::make_pair(solvent_str, new AccT(N,1)) );
        accs.insert( std::make_pair(rs_str, new AccT(2*N,1)) );
        accs.insert( std::make_pair(val_str, new AccT(N,1)) );
        accs.insert( std::make_pair(M_str, new AccT(2*N,N)) );
        accs.insert( std::make_pair(pi_str, new AccT(N,1)) );
    }


    void new_sample(Eigen::MatrixXd Z) {
#if SINGLE_GREEKS
        accs[greeks_str]->new_sample(bsn->get_scalar_allGreeks(Z));
#else
        accs[delta1_str]->new_sample( bsn->get_delta_v1() );
        accs[rho_str]->new_sample(    bsn->get_rho());
        accs[theta_str]->new_sample(  bsn->get_theta(Z));
        accs[vega_str]->new_sample(   bsn->get_vega(Z));
#endif
        accs[assets_str]->new_sample(bsn->get_assets());
        accs[solvent_str]->new_sample(bsn->get_solvent());
        accs[rs_str]->new_sample(bsn->get_rs());
        accs[val_str]->new_sample(bsn->get_valuation());
        accs[M_str]->new_sample(bsn->get_M());
        accs[pi_str]->new_sample(bsn->get_pi());
        samples_count += 1;
    }


    std::unordered_map<std::string, Eigen::MatrixXd> result_object(const int k)
    {
        std::unordered_map<std::string, Eigen::MatrixXd> res;
        auto res_mean = extract(MCUtil::StatType::MEAN);
        auto res_var = extract(MCUtil::StatType::VARIANCE);
        Eigen::MatrixXd count;
        count = Eigen::MatrixXd::Zero(2,1);
        res["Variance " + count_str] = count;
        count(0,0) = 0;
        count(1,0)  = samples_count;
        res[count_str] = count;
        res.insert(res_mean.begin(), res_mean.end());
        res.insert(res_var.begin(), res_var.end());
        return res;
    }


    /*!
     * @brief      Extracts all results for previously registered observers.
     * @param st   Type of statistic which should be extracted. For example MCUtil::StatType::MEAN
     * @return     Returns accumulated result of type T and statistic st
     */
    auto extract_plain(const MCUtil::StatType st) {
        std::vector<double> res;
        for (auto el : accs) {
            LOG(ERROR) << "not implemented yet";
            //res.emplace_back(el.second.extract(st));
        }
        return res;
    }


    /*!
     * @brief      Extracts all results for previously registered observers.
     * @param st   Type of statistic which should be extracted. For example MCUtil::StatType::MEAN
     * @return     Returns pairs of descriptions and accumulated result of type T with statistic st
     */
    std::unordered_map<std::string, Eigen::MatrixXd> extract(const MCUtil::StatType st) {
        std::unordered_map<std::string, Eigen::MatrixXd> res;
        std::string description_prefix = "";
        if(st == MCUtil::StatType::VARIANCE)
            description_prefix = "Variance ";
        for (auto el : accs) {
            if ( !res.insert( std::make_pair( description_prefix + el.first, el.second->extract(st) ) ).second ) {
                LOG(ERROR) << "double result for " << el.first;
            }
        }
        Eigen::MatrixXd count;
        count = Eigen::MatrixXd::Zero(2,1);
        count(0,0) = 0;
        count(1,0)  = samples_count;
        res[description_prefix + count_str] = count;
        return res;
    }

    std::unordered_map<std::string, Eigen::MatrixXd> extract() {
        auto res_m = extract(MCUtil::StatType::MEAN);
        auto res_v = extract(MCUtil::StatType::VARIANCE);
        res_m.insert(res_v.begin(), res_v.end());
        return res_m;
    }
};

#endif //VALUATION_RESULTS_BS_HPP
