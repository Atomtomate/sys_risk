/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "ER_Network.hpp"
void ER_Network::test_init_network() {
    test_init_network(N, p, val, setM, T, r);
}

void ER_Network::test_init_network(const long N_in, const double p_in, const double val_in, const int which_to_set, const double T_new, const double r_new) {
    // ===== Initialization of temporary variables =====
    T = T_new; r = r_new; N = N_in; p = p_in; val = val_in; setM = which_to_set;
    Z.resize(N);
    itSigma.resize(N, N);
    var_h.resize(N);
    S0 = Eigen::VectorXd::Zero(N);
    debt = Eigen::VectorXd::Zero(N);

    // ===== Preparation of Log Normal Dist. =====
    double scale = 0.15; // volatility, higher -> prob to default increases
    Eigen::MatrixXd sigma = scale*scale* Eigen::MatrixXd::Identity(N, N);
    itSigma = (T * sigma).inverse();
    var_h = T * r - T * sigma.diagonal().array() * sigma.diagonal().array() / 2.;

    // ===== creating correlated normal distribution from Eigen Sigma =====
    double *sigma_arr = new double[N * N];
    double sigma2d_arr[N][N];
    Eigen::MatrixXd::Map(sigma_arr, sigma.rows(), sigma.cols()) = sigma;
    memcpy(sigma2d_arr[0], sigma_arr, N*N*sizeof(double));
    Z_dist = trng::correlated_normal_dist<>(&sigma2d_arr[0][0], &sigma2d_arr[N - 1][N - 1] + 1);
    delete[] sigma_arr;
    gen_z.seed(1);



    // ===== Generating Black Scholes Network =====
    S0 = Eigen::VectorXd::Constant(N,1.0).array() + T*sigma.diagonal().array()*sigma.diagonal().array()/2.0;            // <S_t> = S_0 - T*sigma^2/2 => This sets <S_t> ~ 1
    debt = ((var_h + S0).array() - T*r)*Eigen::VectorXd::Constant(N, 1.0/(1.0-val)).array();
    if(bsn != nullptr)
        delete bsn;
    LOG(TRACE) << "creating new Black Scholes Network";
    bsn = new BlackScholesNetwork(T, r);
    LOG(TRACE) << "Initializing random connectivity matrix";
    init_M_ER(p, val, which_to_set);
    LOG(TRACE) << "Generating Model";

    initialized = true;
}

std::unordered_map<std::string, Eigen::MatrixXd> ER_Network::test_ER_valuation(const long N_in, const long N_Samples, const long N_networks) {
    test_init_network();

    const std::string rs_str("RS");
    const std::string M_str("M");
    const std::string assets_str("Assets");
    const std::string solvent_str("Solvent");
    const std::string val_str("Valuation");
    const std::string delta1_str("Delta using Jacobians");
    const std::string delta2_str("Delta using Log");
    std::unordered_map<std::string, Eigen::MatrixXd> res;

    auto f_dist = [this]() -> Eigen::MatrixXd { return this->draw_from_dist(); };
    auto f_run =  [this](const Eigen::Ref<const Eigen::MatrixXd>& x) {this->run(x); };

    // ===== Defining observables =====
    auto asset_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_assets(); };
    auto rs_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_rs(); };
    auto M_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_M(); };
    auto sol_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_solvent(); };
    auto delta_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_delta_v1();};
    std::function<const Eigen::MatrixXd(void)> assets_obs(std::ref(asset_obs_lambda));
    S.register_observer(assets_obs, assets_str, N, 1);
    std::function<const Eigen::MatrixXd(void)> rs_obs(std::ref(rs_obs_lambda));
    S.register_observer(rs_obs, rs_str, 2*N, 1);
    std::function<const Eigen::MatrixXd(void)> M_obs(std::ref(M_obs_lambda));
    S.register_observer(M_obs, M_str, N, 2*N);
    std::function<const Eigen::MatrixXd(void)> sol_obs(std::cref(sol_obs_lambda));
    S.register_observer(sol_obs, solvent_str, N, 1);
    //std::function<const Eigen::MatrixXd(void)> valuation_obs = [this]() -> Eigen::MatrixXd { return bsn->get_valuation(); };
    //S.register_observer(valuation_obs, val_str, N, 1);
    std::function<const Eigen::MatrixXd(void)> deltav1_obs(std::cref(delta_obs_lambda));
    S.register_observer(deltav1_obs, delta1_str, 2 * N , N);
    //std::function<const Eigen::MatrixXd(void)> deltav2_obs   = [this]() -> Eigen::MatrixXd { return this->delta_v2();};
    //S.register_observer(deltav2_obs, delta2_str, 2 * N , N);
    //std::function<const Eigen::MatrixXd(void)> out_obs =  [this]() -> Eigen::MatrixXd { return this->test_out();};
    //S.register_observer(out_obs, "Debug Out" ,1, 1);

    std::cout << "Running Valuation for N = " << N <<  ", p =" << p << ", sum_j M_ij = " << val << "\n";
    std::cout << "Preparing to run"<< std::flush ;
    for(int jj = 0; jj < N_networks; jj++)
    {
        std::cout << "\r  ---> " << 100.0*static_cast<double>(jj)/N_networks << "% of runs finished" <<std::flush;
        init_M_ER(p, val, setM);
        if(!initialized) LOG(ERROR) << "attempting to run uninitialized network";
        S.draw_samples(f_run, f_dist, N_Samples);
    }
    std::cout << "\r" << std::endl;

    //LOG(ERROR) << "M: \n" << bsn->get_M();
    //LOG(WARNING) << "=========================================";
    //LOG(ERROR) << "debt: \n" << debt;
    //@TODO: workaround for pybind11. A sane version of this would only use maps
    auto res_mean = S.extract(MCUtil::StatType::MEAN);
    auto res_var = S.extract(MCUtil::StatType::VARIANCE);
    for (auto el : res_mean) {
        if(el.first.compare(rs_str) == 0){ mean_rs = el.second; res[rs_str] = el.second;}
        else if(el.first.compare(M_str) == 0){ mean_M = el.second; res[M_str] = el.second;}
        else if(el.first.compare(assets_str) == 0){ mean_assets = el.second; res[assets_str] = el.second;}
        else if(el.first.compare(solvent_str) == 0){ mean_solvent = el.second; res[solvent_str] = el.second;}
        else if(el.first.compare(val_str) == 0){ mean_valuation = el.second; res[val_str] = el.second;}
        else if(el.first.compare(delta1_str) == 0){ mean_delta_jac = el.second; res[delta1_str] = el.second;}
        else if(el.first.compare(delta2_str) == 0){ mean_delta_log = el.second; res[delta2_str] = el.second;}
        else LOG(WARNING) << "result " << el.first << ", not saved";
    }
    for (auto el : res_var) {
        if(el.first.compare(rs_str) == 0){ var_rs = el.second; res["Variance "+rs_str] = el.second;}
        else if(el.first.compare(M_str) == 0){ var_M = el.second; res["Variance "+M_str] = el.second;}
        else if(el.first.compare(assets_str) == 0){ var_assets = el.second; res["Variance "+assets_str] = el.second;}
        else if(el.first.compare(solvent_str) == 0){ var_solvent = el.second; res["Variance "+solvent_str] = el.second;}
        else if(el.first.compare(val_str) == 0){ var_valuation = el.second; res["Variance "+val_str] = el.second;}
        else if(el.first.compare(delta1_str) == 0){ var_delta_jac= el.second; res["Variance "+delta1_str] = el.second;}
        else if(el.first.compare(delta2_str) == 0){ var_delta_log= el.second; res["Variance "+delta2_str] = el.second;}
        else LOG(WARNING) << "result " << el.first << ", not saved";
    }

    initialized = false;
    return res;
}

void ER_Network::init_M_ER(const double p, const double val, int which_to_set)
{
    if(val < 0 || val > 1) throw std::logic_error("Value is not in [0,1]");
    if(p < 0 || p > 1) throw std::logic_error("p is not a probability");
    connectivity = N*p;
    int i = 0;
    Eigen::MatrixXd M;
    do {
        M = Eigen::MatrixXd::Zero(N, 2*N);
        //@TODO: use bin. dist. to generate vectorized,
        //@TODO: this is terrifingly inefficient
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = i + 1; j < N; j++) {
                if (which_to_set == 1 || which_to_set == 0) {
                    if (u_dist(gen_u) < p)
                        M(i, j) = 1.0;
                    if (u_dist(gen_u) < p)
                        M(j, i) = 1.0;
                }
                if (which_to_set == 2 || which_to_set == 0) {
                    if (u_dist(gen_u) < p)
                        M(i, j + N) = 1.0;
                    if (u_dist(gen_u) < p)
                        M(j, i + N) = 1.0;
                }
            }
        }
        //@TODO: implement S sums
        auto sum_d_j = M.rightCols(N).rowwise().sum();
        auto sum_s_j = M.leftCols(N).rowwise().sum();
        auto sum_i = M.colwise().sum();
        //for(int ii = 0; ii < 2*N; ii++)
        //{
        //    if(sum_i(ii) > 0)
        //       M.col(ii) = M.col(ii)/sum_i(ii);
        //}
        for(int ii = 0; ii < N; ii++)
        {
            if(sum_d_j(ii) > 0)
                M.row(ii) = (val/sum_d_j(ii))*M.row(ii);
        }
        if(i > 50000)
        {
            Eigen::IOFormat CleanFmt(2, 0, " ", "\n", "[", "]");
            LOG(WARNING)<<"\n" << M.format(CleanFmt) << "\n\n";
            LOG(ERROR) << M.colwise().sum();
            throw std::runtime_error("\n\nToo many rejections during generation of M!\n\n");
        }
        i++;
    }while(M.colwise().sum().maxCoeff() >= 1);
    if(i > 500) LOG(WARNING) << "\rrejected " << i << " candidates for network matrix";
    bsn->re_init(M, S0, debt);
}


const Eigen::MatrixXd ER_Network::delta_v2() {
    //delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
    Eigen::VectorXd ln_fac = (itSigma * Z).array() / (bsn->get_S0()).array();
    Eigen::MatrixXd m = std::exp(-r * T) * (ln_fac * (bsn->get_rs()).transpose()).transpose();
    //Eigen::MatrixXd::Map(&res[0], m.rows(), m.cols()) = m;
    return m;
}


const Eigen::MatrixXd ER_Network::draw_from_dist() {
    for (long d = 0; d < N; d++) {
        Z(d) = Z_dist(gen_z);
    }
    Eigen::VectorXd S_log = var_h + std::sqrt(T) * Z;
    LOG(ERROR) << "var_h:\n" << var_h << "\n sqrt(T): " << std::sqrt(T)<< "\nS_log: \n" << S_log;
    return S_log.array().exp();
}
