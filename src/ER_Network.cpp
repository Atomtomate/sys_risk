/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "ER_Network.hpp"
void ER_Network::test_init_network() {
    test_init_network(N, p, val_row, val_col, setM, T, r);
}

void ER_Network::test_init_network(const long N_in, const double p_in, const double val_row_in, const double val_col_in, const int which_to_set, const double T_new, const double r_new, const double default_prob_scale) {
    // ===== Initialization of temporary variables =====
    T = T_new; r = r_new; N = N_in; p = p_in; val_row = val_row_in; val_col = val_col_in; setM = which_to_set;
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
    debt = ((var_h + S0).array() - T*r)*Eigen::VectorXd::Constant(N, default_prob_scale/(1.0-val_row)).array();
    if(bsn != nullptr)
        delete bsn;
    LOG(TRACE) << "Generating Model";
    LOG(TRACE) << "creating new Black Scholes Network";
    bsn = new BlackScholesNetwork(T, r);
    LOG(TRACE) << "Initializing random connectivity matrix";
    try{
        init_M_ER(p, val_row, val_col, which_to_set);
        initialized = true;
    }
    catch (const std::runtime_error& e)
    {
        LOG(ERROR) << "unable to create ER Model graph for N=" << N << ", p=" << p << ", val_row="<< val_row;
        initialized = false;
    }

}


std::unordered_map<std::string, Eigen::MatrixXd> ER_Network::test_ER_valuation(const long N_Samples, const long N_networks) {
    test_init_network();

    const std::string count_str("#Samples");
    const std::string rs_str("RS");
    const std::string M_str("M");
    const std::string assets_str("Assets");
    const std::string solvent_str("Solvent");
    const std::string val_str("Valuation");
    const std::string delta1_str("Delta using Jacobians");
    const std::string delta2_str("Delta using Log");
    const std::string io_deg_str("In/Out degree distribution");
    std::unordered_map<std::string, Eigen::MatrixXd> res;

    auto f_dist = [this]() -> Eigen::MatrixXd { return this->draw_from_dist(); };
    auto f_run =  [this](const Eigen::Ref<const Eigen::MatrixXd>& x) {this->run(x); };

    // ===== Defining observables =====
    auto asset_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_assets(); };
    auto rs_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_rs(); };
    auto M_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_M(); };
    auto sol_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_solvent(); };
    auto delta_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_delta_v1();};
    auto io_deg_obs_lambda = [this]() -> Eigen::MatrixXd { return this->io_deg_dist; };
    if(S != nullptr)
        delete S;
    S = new MCUtil::Sampler<Eigen::MatrixXd>();
    std::function<const Eigen::MatrixXd(void)> assets_obs(std::cref(asset_obs_lambda));
    S->register_observer(assets_obs, assets_str, N, 1);
    std::function<const Eigen::MatrixXd(void)> rs_obs(std::cref(rs_obs_lambda));
    S->register_observer(rs_obs, rs_str, 2*N, 1);
    std::function<const Eigen::MatrixXd(void)> M_obs(std::cref(M_obs_lambda));
    S->register_observer(M_obs, M_str, N, 2*N);
    std::function<const Eigen::MatrixXd(void)> sol_obs(std::cref(sol_obs_lambda));
    S->register_observer(sol_obs, solvent_str, N, 1);
    std::function<const Eigen::MatrixXd(void)> deltav1_obs(std::cref(delta_obs_lambda));
    S->register_observer(deltav1_obs, delta1_str, 2 * N , N);
    std::function<const Eigen::MatrixXd(void)> io_deg_obs(std::cref(io_deg_obs_lambda));
    S->register_observer(io_deg_obs, io_deg_str, 2 , N);

    //std::function<const Eigen::MatrixXd(void)> valuation_obs = [this]() -> Eigen::MatrixXd { return bsn->get_valuation(); };
    //S->register_observer(valuation_obs, val_str, N, 1);
    //std::function<const Eigen::MatrixXd(void)> deltav2_obs   = [this]() -> Eigen::MatrixXd { return this->delta_v2();};
    //S->register_observer(deltav2_obs, delta2_str, 2 * N , N);
    //std::function<const Eigen::MatrixXd(void)> out_obs =  [this]() -> Eigen::MatrixXd { return this->test_out();};
    //S->register_observer(out_obs, "Debug Out" ,1, 1);

    std::cout << "Running Valuation for N = " << N <<  ", p =" << p << ", sum_j M_ij = " << val_row << ", sum_i M_ij = " << val_col  << "\n";
    std::cout << "Preparing to run"<< std::flush ;
    for(int jj = 0; jj < N_networks; jj++)
    {
        std::cout << "\r  ---> " << 100.0*static_cast<double>(jj)/N_networks << "% of runs finished" <<std::flush;
        try{
            init_M_ER(p, val_row, val_col, setM);
            S->draw_samples(f_run, f_dist, N_Samples);
        } catch (const std::runtime_error& e)
        {
            LOG(ERROR) << "Skipping uninitialized network for N = " << N << ", p=" << p << ", val_row=" << val_row << ", val_col="<<val_col;
            initialized = 0;
        }
    }
    std::cout << "\r" << std::endl;

    auto res_mean = S->extract(MCUtil::StatType::MEAN);
    auto res_var = S->extract(MCUtil::StatType::VARIANCE);
    count = Eigen::MatrixXd::Zero(2,1);
    res["Variance " + count_str] = count;
    count(0,0) = N_Samples*N_networks;
    count(1,0)  = S->get_count();
    res[count_str] = count;
    for (auto el : res_mean) {
        if(el.first.compare(rs_str) == 0){ mean_rs = el.second; res[rs_str] = el.second;}
        else if(el.first.compare(M_str) == 0){ mean_M = el.second; res[M_str] = el.second;}
        else if(el.first.compare(assets_str) == 0){ mean_assets = el.second; res[assets_str] = el.second;}
        else if(el.first.compare(solvent_str) == 0){ mean_solvent = el.second; res[solvent_str] = el.second;}
        else if(el.first.compare(val_str) == 0){ mean_valuation = el.second; res[val_str] = el.second;}
        else if(el.first.compare(delta1_str) == 0){ mean_delta_jac = el.second; res[delta1_str] = el.second;}
        else if(el.first.compare(delta2_str) == 0){ mean_delta_log = el.second; res[delta2_str] = el.second;}
        else if(el.first.compare(io_deg_str) == 0){ mean_io_deg_dist = el.second; res[io_deg_str] = el.second;}
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
        else if(el.first.compare(io_deg_str) == 0){ var_io_deg_dist = el.second; res["Variance" + io_deg_str] = el.second;}
        else LOG(WARNING) << "result " << el.first << ", not saved";
    }

    initialized = false;
    return res;
}

void ER_Network::init_M_ER(const double p, const double val_row, const double val_col, int which_to_set) {

    if (val_row < 0 || val_row > 1) throw std::logic_error("Row sum is not in [0,1]");
    if (val_col < 0 || val_col > 1) throw std::logic_error("Col sum is not in [0,1]");
    if (p < 0 || p > 1) throw std::logic_error("p is not a probability");
    connectivity = N * p;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, 2 * N);

    //Utils::gen_sinkhorn(&M, gen_u, p, val_row, val_col, which_to_set);
    int degree = std::floor(p * N);

        Utils::gen_fixed_degree(&M, gen_u, degree, val_col, which_to_set);
        io_deg_dist = Utils::in_out_degree(&M);

    /*LOG(INFO) << "Using rejection sampling: ";
    try{
        Utils::gen_basic_rejection(&M, gen_u, p, val_row, val_col, which_to_set);
        LOG(INFO) << in_out_degree(&M);
    }catch (const std::runtime_error& e)
    {
        LOG(INFO) <<"rejection sampler failed";
    }
    auto io_deg = in_out_degree(&M);
    LOG(INFO) << io_deg;
    double in_avg = 0.;
    double out_avg = 0.;
    for(int i = 0; i < io_deg.cols(); i++)
    {
        in_avg += i*io_deg(0,i);
        out_avg += i*io_deg(1,i);
    }
    in_avg /= io_deg.cols();
    out_avg /= io_deg.cols();
    LOG(INFO) << "avg in degree: " << in_avg << ", avg out degree: " << out_avg;

    LOG(INFO) << "Using Sinkhorn Algorithm: ";
    exit(0);
    */

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
    //LOG(ERROR) << "var_h:\n" << var_h << "\n sqrt(T): " << std::sqrt(T)<< "\nS_log: \n" << S_log;
    return S_log.array().exp();
}

