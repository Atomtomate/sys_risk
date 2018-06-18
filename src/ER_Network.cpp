/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "ER_Network.hpp"

void ER_Network::init_network(unsigned int N_in, double p_in, double val_in, unsigned int which_to_set) {
    init_network(N_in, p_in, val_in, which_to_set, this->T, this->r);
}

void ER_Network::init_network(const unsigned int N_in, const double p_in, const double val_in, const unsigned int which_to_set, const double T_new, const double r_new) {
    T = T_new; r = r_new; N = N_in; p = p_in; val = val_in; setM = which_to_set;
    Z.resize(N);

    Eigen::MatrixXd S0 = Eigen::VectorXd::Constant(N, 1.0);
    Eigen::MatrixXd debt = Eigen::VectorXd::Constant(N, 1.1);
    itSigma.resize(N, N);
    var_h.resize(N);

    //BlackScholesNetwork::BlackScholesNetwork(Eigen::MatrixXd& M, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r):
    if(bsn != nullptr)
        delete bsn;
    LOG(TRACE) << "creating new Black Scholes Network";
    bsn = new BlackScholesNetwork(T, r);
    LOG(TRACE) << "Initializing random connectivity matrix";
    init_M_ER(p, val, which_to_set, S0, debt);
    LOG(TRACE) << "Generating Model";


    double scale = 0.15; // volatility, higher -> prob to default increases
    // random asset stuff
    Eigen::MatrixXd sigma = scale*scale* T * Eigen::MatrixXd::Identity(N, N);
    itSigma = (T * sigma).inverse();
    var_h = T * r - T * sigma.diagonal().array() * sigma.diagonal().array() / 2.;

    // creating correlated normal distribution from Eigen Sigma
    double *sigma_arr = new double[N * N];
    double sigma2d_arr[N][ N];
    Eigen::MatrixXd::Map(sigma_arr, sigma.rows(), sigma.cols()) = sigma;
    //double (*sigma_mat)[N] = (double (*)[N]) sigma_arr;
    memcpy(sigma2d_arr[0], sigma_arr, N*N*sizeof(double));
    Z_dist = trng::correlated_normal_dist<>(&sigma2d_arr[0][0], &sigma2d_arr[N - 1][N - 1] + 1);
    delete[] sigma_arr;
    gen_z.seed(1);
    initialized = true;
}


std::unordered_map<std::string, Eigen::MatrixXd> ER_Network::test_ER_valuation(const long N_in, const long N_Samples, const long N_networks) {
    init_network(N_in, p, val, setM);

    const std::string rs_str("RS");
    const std::string assets_str("Assets");
    const std::string solvent_str("Solvent");
    const std::string val_str("Valuation");
    const std::string delta1_str("Delta using Jacobians");
    std::unordered_map<std::string, Eigen::MatrixXd> res;

    //@TODO: use lambdas instead of bind, it is not 2011....
    //auto f_dist2 = [this]()-> std::result_of_t<decltype(&ER_Network::draw_from_dist)(ER_Network)> { return this->draw_from_dist(); };
    auto f_dist = [this]() -> Eigen::MatrixXd { return this->draw_from_dist(); };//std::bind(&ER_Network::draw_from_dist, this);
    auto f_run =  [this](const Eigen::Ref<const Eigen::MatrixXd>& x) {this->run(x); };//std::bind(&ER_Network::run, this, std::placeholders::_1);
    std::function<const Eigen::MatrixXd(void)> assets_obs    = [this]() -> Eigen::MatrixXd { return bsn->get_assets(); };//std::bind(&BlackScholesNetwork::get_assets, bsn);
    std::function<const Eigen::MatrixXd(void)> rs_obs        = [this]() -> Eigen::MatrixXd { return bsn->get_rs(); };//std::bind(&BlackScholesNetwork::get_rs, bsn);
    std::function<const Eigen::MatrixXd(void)> sol_obs       = [this]() -> Eigen::MatrixXd { return bsn->get_solvent(); };
    std::function<const Eigen::MatrixXd(void)> valuation_obs = [this]() -> Eigen::MatrixXd { return bsn->get_valuation(); };//std::bind(&BlackScholesNetwork::get_valuation, bsn);
    std::function<const Eigen::MatrixXd(void)> deltav1_obs   = [this]() -> Eigen::MatrixXd { return bsn->get_delta_v1();};//std::bind(&BlackScholesNetwork::get_delta_v1, bsn);
    std::function<const Eigen::MatrixXd(void)> deltav2_obs   = [this]() -> Eigen::MatrixXd { return this->delta_v2();};//std::bind(&ER_Network::delta_v2, this);
    //std::function<std::vector<double>(void)> out_obs = std::bind(&ER_Network::test_out, this);

    // usage: register std::function with no parameters and boost::accumulator compatible return value. 2nd,... parameters are used to construct accumulator
    S.register_observer(rs_obs, rs_str, 2*N, 1);
    S.register_observer(assets_obs, assets_str, N, 1);
    S.register_observer(sol_obs, solvent_str, N, 1);
    S.register_observer(valuation_obs, val_str, N, 1);
    //S.register_observer(std::bind(&ER_Network::sumM, this), "Sum over M", 1);
    //S.register_observer(out_obs, "Debug Out" ,1);

    S.register_observer(deltav1_obs, delta1_str, 2 * N , N);
    //S.register_observer(deltav2_obs, "Delta using Log", 2 * N , N);
    std::cout << "Running Valuation for N = " << N <<  ", p =" << p << "\n";

    std::cout << "Preparing to run"<< std::flush ;
    for(int jj = 0; jj < N_networks; jj++)
    {
        std::cout << "\r  ---> " << 100.0*static_cast<double>(jj)/N_networks << "% of runs finished" <<std::flush;
        init_M_ER(p, val, setM, Eigen::VectorXd::Constant(N, 1.0), Eigen::VectorXd::Constant(N, 1.1));
        if(!initialized) LOG(ERROR) << "attempting to run uninitialized network";
        S.draw_samples(f_run, f_dist, N_Samples);
    }
    std::cout << "\r";
    //@TODO: workaround for pybind11. A sane version of this would only use maps
    auto res_mean = S.extract(MCUtil::StatType::MEAN);
    auto res_var = S.extract(MCUtil::StatType::VARIANCE);
    for (auto el : res_mean) {
        if(el.first.compare(rs_str) == 0){ mean_rs = el.second; res[rs_str] = el.second;}
        else if(el.first.compare(assets_str) == 0){ mean_assets = el.second; res[assets_str] = el.second;}
        else if(el.first.compare(solvent_str) == 0){ mean_solvent = el.second; res[solvent_str] = el.second;}
        else if(el.first.compare(val_str) == 0){ mean_valuation = el.second; res[val_str] = el.second;}
        else if(el.first.compare(delta1_str) == 0){ mean_delta_jac = el.second; res[delta1_str] = el.second;}
        else LOG(WARNING) << "result " << el.first << ", not saved";
    }
    for (auto el : res_var) {
        if(el.first.compare(rs_str) == 0) var_rs = el.second;
        else if(el.first.compare(assets_str) == 0) var_assets = el.second;
        else if(el.first.compare(solvent_str) == 0) var_solvent = el.second;
        else if(el.first.compare(val_str) == 0) var_valuation = el.second;
        else if(el.first.compare(delta1_str) == 0) var_delta_jac= el.second;
        else LOG(WARNING) << "result " << el.first << ", not saved";
    }

    initialized = false;
    return res;
}

const Eigen::MatrixXd ER_Network::draw_from_dist() {
    Eigen::VectorXd S_log(N);             // log of lognormal distribution exogenous assets, without a_0
    for (unsigned int d = 0; d < N; d++) {
        Z(d) = Z_dist(gen_z);
    }
    S_log = var_h + std::sqrt(T) * Z;
    //std::vector<double> res;
    //res.resize(S_log.size());
    //Eigen::VectorXd::Map(&res[0], S_log.size()) = S_log.array().exp();
    return S_log.array().exp();
}

const Eigen::MatrixXd ER_Network::delta_v2() {
    //delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
    //std::vector<double> res(2 * N * N);
    Eigen::VectorXd ln_fac = (itSigma * Z).array() / (bsn->get_S0()).array();
    Eigen::MatrixXd m = std::exp(-r * T) * (ln_fac * (bsn->get_rs()).transpose()).transpose();
    //Eigen::MatrixXd::Map(&res[0], m.rows(), m.cols()) = m;
    return m;
}

void ER_Network::init_M_ER(const double p, const double val, unsigned int which_to_set, const Eigen::VectorXd& s0, const Eigen::VectorXd& debt)
{
    if(val < 0 || val > 1) throw std::logic_error("Value is not in [0,1]");
    if(p < 0 || p > 1) throw std::logic_error("p is not a probability");
    connectivity = N*p;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, 2*N);

    //@TODO: use bin. dist. to generate vectorized
    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = i + 1; j < N; j++)
        {
            if(which_to_set == 1 || which_to_set == 0)
            {
                if(u_dist(gen_u) < p)
                    M(i,j) = 1.0;
                if(u_dist(gen_u) < p)
                    M(j,i) = 1.0;
            }
            if(which_to_set == 2 || which_to_set == 0)
            {
                if(u_dist(gen_u) < p)
                    M(i,j+N) = 1.0;
                if(u_dist(gen_u) < p)
                    M(j,i+N) = 1.0;
            }
        }
    }
    //@TODO: valid normalization
    auto col_sum = M.leftCols(N).colwise().sum();
    auto row_r_sum = M.leftCols(N).rowwise().sum();
    auto row_s_sum = M.rightCols(N).rowwise().sum();
    double max = std::max({col_sum.maxCoeff(), row_r_sum.maxCoeff(), row_s_sum.maxCoeff()});
    //LOG(ERROR) << "M:\n" << M << "\nCol Sum: \n" <<col_sum << std::endl <<  std::endl << max << std::endl;
    M = (val/max)*M;
    //LOG(ERROR) << M;
    bsn->re_init(M, s0, debt);
}
