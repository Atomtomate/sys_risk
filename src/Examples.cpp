//
// Created by julian on 5/9/18.
//

#include "Examples.hpp"


void N2_network::test_N2_valuation()
{
    //@TODO: acc std::vector
    Sampler<std::vector<double>> S;
    const int N = 2;


    auto f_dist = std::bind(&N2_network::draw_from_dist, this);
    auto f_run = std::bind(&N2_network::run, this, std::placeholders::_1);
    std::function<std::vector<double>(void)> assets_obs = std::bind(&BlackScholesNetwork::get_assets, &bsn);
    std::function<std::vector<double>(void)> rs_obs = std::bind(&BlackScholesNetwork::get_rs, &bsn);
    std::function<std::vector<double>(void)> sol_obs = std::bind(&BlackScholesNetwork::get_solvent, &bsn);
    std::function<std::vector<double>(void)> valuation_obs = std::bind(&BlackScholesNetwork::get_valuation, &bsn);

    // usage: register std::function with no parameters and boost::accumulator compatible return value. 2nd,... parameters are used to construct accumulator
    //S.register_observer(assets_obs, 2);
    //S.register_observer(rs_obs, 4);
    //S.register_observer(sol_obs, 2);
    //S.register_observer(valuation_obs, 2);
    //S.register_observer(std::bind(&N2_network::test_out, &n2nw), 1);
    S.register_observer(std::bind(&BlackScholesNetwork::get_delta_v1, &bsn), 2*N*N);
    S.draw_samples(f_run, f_dist, 10);
    auto res = S.get_acc();
}

Eigen::VectorXd run_N2_network()
{

    // setting values for example calculation

    // ---------------------------------------------


    // running simulation
    // --- setup


    // --- run
    //-------------------  auto v_res = bsn.get_solvent();
    //-------------------  auto solvent = bsn.get_solvent();

    //v_res = rs.head(N) + rs.tail(N);
    //Eigen::VectorXi solvent = classify_solvent(v_res, debt);

    // --- extract
    //auto Jrs_m1 = bsn.iJacobian_fx();
    //Eigen::MatrixXd Jrs_m1 = (eye22 - jacobian_rs(M, solvent)).inverse();
    //Eigen::MatrixXd Ja = jacobian_a(solvent);
    //Eigen::VectorXd tmp = (S0.array()*St.array()).log() - S0.array().log() - var_h.array();

    //rs_acc += rs;

    // prepare output
    //V_out = (eye2 - M)*rs;
    //for(unsigned int j = 0; j < N; j++)
    //{
    //    ss << v_res(j) << "\t";
    //}
    //ss << static_cast<std::underlying_type<ACase>::type>(classify(v_res, debt)) << std::endl;

    /*
    std::cout << ss.str();
    rs_acc = rs_acc/nPoints;
    v_res = rs_acc.head(N) + rs_acc.tail(N);
    LOG(INFO) << "delta_pw: \n" << delta_pw/nPoints;
    LOG(INFO) << "delta_lg: \n" << delta_lg/nPoints;
    LOG(INFO) << "rs: \n" << rs_acc;
    LOG(INFO) << "v: \n" << v_res;
    */
}
