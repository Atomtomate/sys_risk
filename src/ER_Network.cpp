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
    T = T_new;
    r = r_new;
    N = N_in;
    p = p_in;
    val = val_in;
    setM = which_to_set;
    Z.resize(N);
    Eigen::MatrixXd S0 = Eigen::VectorXd::Constant(N, 1.0);
    Eigen::MatrixXd debt = Eigen::VectorXd::Constant(N, 11.3);
    itSigma.resize(N, N);
    var_h.resize(N);

    //BlackScholesNetwork::BlackScholesNetwork(Eigen::MatrixXd& M, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r):
    if(bsn != nullptr)
        delete bsn;
    bsn = new BlackScholesNetwork(T, r);
    init_M_ER(p, val, which_to_set, S0, debt);


    // random asset stuff
    Eigen::MatrixXd sigma = T * Eigen::MatrixXd::Identity(N, N);
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
}


void ER_Network::test_ER_valuation(const unsigned int N_in, const unsigned int N_Samples) {
    init_network(N_in, p, val, setM);

    //@TODO: use lambdas instead of bind, it is not 2011....
    auto f_dist = std::bind(&ER_Network::draw_from_dist, this);
    auto f_run = std::bind(&ER_Network::run, this, std::placeholders::_1);
    std::function<const Eigen::MatrixXd(void)> assets_obs = std::bind(&BlackScholesNetwork::get_assets, bsn);
    //std::function<std::vector<double>(void)> rs_obs = std::bind(&BlackScholesNetwork::get_rs, bsn);
    std::function<const Eigen::MatrixXd(void)> sol_obs = std::bind(&BlackScholesNetwork::get_solvent, bsn);
    std::function<const Eigen::MatrixXd(void)> valuation_obs = std::bind(&BlackScholesNetwork::get_valuation, bsn);
    std::function<const Eigen::MatrixXd(void)> deltav1_obs = std::bind(&BlackScholesNetwork::get_delta_v1, bsn);
    std::function<const Eigen::MatrixXd(void)> deltav2_obs = std::bind(&ER_Network::delta_v2, this);
    //std::function<std::vector<double>(void)> out_obs = std::bind(&ER_Network::test_out, this);

    // usage: register std::function with no parameters and boost::accumulator compatible return value. 2nd,... parameters are used to construct accumulator
    //S.register_observer(rs_obs, 2*N);
    S.register_observer(assets_obs, "Assets", N, 1);
    S.register_observer(sol_obs, "Solvent", N, 1);
    S.register_observer(valuation_obs, "Valuation", N, 1);
    //S.register_observer(std::bind(&ER_Network::sumM, this), "Sum over M", 1);
    //S.register_observer(out_obs, "Debug Out" ,1);

    S.register_observer(deltav1_obs, "Delta using Jacobians", 2 * N , N);
    S.register_observer(deltav2_obs, "Delta using Log", 2 * N , N);
    LOG(INFO) << "Running Valuation for N = " << N;


    S.draw_samples(f_run, f_dist, N_Samples);
    LOG(INFO) << std::endl << " ======= Means ======= ";


    auto res = S.extract(MCUtil::StatType::MEAN);
    for (auto el : res) {
        std::cout << el.first << ": " << std::endl;
        std::cout << el.second;
        std::cout << std::endl << std::endl;
    }

    LOG(INFO) << std::endl << " ======= Vars ======= ";
    auto res_var = S.extract(MCUtil::StatType::VARIANCE);
    for (auto el : res_var) {
        std::cout << el.first << ": " << std::endl;
        std::cout << el.second;
        std::cout << std::endl << std::endl;
    }
}

Eigen::MatrixXd ER_Network::draw_from_dist() {
    Eigen::VectorXd S_log(N);             // log of lognormal distribution exogenous assets, without a_0
    for (unsigned int d = 0; d < N; d++) {
        Z(d) = Z_dist(gen_z);
        S_log(d) = var_h(d) + std::sqrt(T) * Z(d);
    }
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
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, 2*N);
    trng::yarn2 gen_u;
    trng::uniform01_dist<> u_dist;
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
