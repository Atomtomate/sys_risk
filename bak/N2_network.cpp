/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "N2_network.hpp"


void N2_network::test_N2_valuation() {
    //@TODO: acc std::vector
    MCUtil::Sampler<std::vector<double>> S;
    const int N = 2;


    auto f_dist = std::bind(&N2_network::draw_from_dist, this);
    auto f_run = std::bind(&N2_network::run, this, std::placeholders::_1);
    /*

    std::function<std::vector<double>(void)> assets_obs = std::bind(&BlackScholesNetwork::get_assets, &bsn);
    std::function<std::vector<double>(void)> rs_obs = std::bind(&BlackScholesNetwork::get_rs, &bsn);
    std::function<std::vector<double>(void)> sol_obs = std::bind(&BlackScholesNetwork::get_solvent, &bsn);
    std::function<std::vector<double>(void)> valuation_obs = std::bind(&BlackScholesNetwork::get_valuation, &bsn);
    std::function<std::vector<double>(void)> deltav1_obs = std::bind(&BlackScholesNetwork::get_delta_v1, &bsn);
    std::function<std::vector<double>(void)> deltav2_obs = std::bind(&N2_network::delta_v2, this);

    // usage: register std::function with no parameters and boost::accumulator compatible return value. 2nd,... parameters are used to construct accumulator
    //S.register_observer(assets_obs, 2);
    //S.register_observer(rs_obs, 4);
    //S.register_observer(sol_obs, 2);
    /*S.register_observer(valuation_obs, "Valuation", 2);
    //std::function<std::vector<double>(void)> out_obs = std::bind(&N2_network::test_out, this);
    //S.register_observer(out_obs, 1);

    S.register_observer(deltav1_obs, "Delta using Jacobians", 8);
    S.register_observer(deltav2_obs, "Delta using Log", 2 * N * N);
    S.draw_samples(f_run, f_dist, 1000);
     */
    LOG(INFO) << "Means: ";
    auto res = S.extract(MCUtil::StatType::MEAN);
    for (auto el : res) {
        Eigen::MatrixXd m;
        if (el.second.size() > N) {
            m = Eigen::MatrixXd::Map(&el.second[0], 2 * N, N);
            LOG(INFO) << el.first << ": " << m;
        }
    }
    LOG(INFO) << "Vars: ";
    auto res_var = S.extract(MCUtil::StatType::VARIANCE);
    for (auto el : res_var) {
        Eigen::MatrixXd m;
        if (el.second.size() > N) {
            m = Eigen::MatrixXd::Map(&el.second[0], 2 * N, N);
            LOG(INFO) << el.first << ": " << m;
        }
    }
}
