//
// Created by julian on 5/21/18.
//

#include "ER_Network.hpp"

void ER_Network::init_network(unsigned int N_in, double p_in, double val_in, char which_to_set) {
    N = N_in;
    p = p_in;
    val = val_in;
    setM = which_to_set;
    Z = Eigen::VectorXd(N);
    Eigen::MatrixXd S0 = Eigen::VectorXd::Constant(N, 1.0);
    Eigen::MatrixXd debt = Eigen::VectorXd::Constant(N, 11.3);
    Eigen::MatrixXd M = Eigen::MatrixXd(N, 2 * N);
    itSigma = Eigen::MatrixXd(N, N);
    var_h = Eigen::VectorXd(N);

    bsn.set_M(M);
    bsn.set_S0(S0);
    bsn.set_debt(debt);

    set_M_ER(p, val, which_to_set);

    // random asset stuff
    Eigen::MatrixXd sigma = T * Eigen::MatrixXd::Identity(N, N);
    itSigma = (T * sigma).inverse();
    var_h = T * r - T * sigma.diagonal().array() * sigma.diagonal().array() / 2.;

    // creating correlated normal distribution from Eigen Sigma
    double *sigma_arr = new double[N * N];
    Eigen::MatrixXd::Map(sigma_arr, sigma.rows(), sigma.cols()) = sigma;
    double (*sigma_mat)[N] = (double (*)[N]) sigma_arr;
    Z_dist = trng::correlated_normal_dist<>(&sigma_mat[0][0], &sigma_mat[N - 1][N - 1] + 1);
    delete sigma_arr;
    gen_z.seed(1);
}


void ER_Network::test_ER_valuation(const unsigned int N_in) {
    init_network(N_in, p, val, setM);
    //@TODO: acc std::vector
    MCUtil::Sampler<std::vector<double>> S;

    auto f_dist = std::bind(&ER_Network::draw_from_dist, this);
    auto f_run = std::bind(&ER_Network::run, this, std::placeholders::_1);
    std::function<std::vector<double>(void)> assets_obs = std::bind(&BlackScholesNetwork::get_assets, &bsn);
    std::function<std::vector<double>(void)> rs_obs = std::bind(&BlackScholesNetwork::get_rs, &bsn);
    std::function<std::vector<double>(void)> sol_obs = std::bind(&BlackScholesNetwork::get_solvent, &bsn);
    std::function<std::vector<double>(void)> valuation_obs = std::bind(&BlackScholesNetwork::get_valuation, &bsn);
    std::function<std::vector<double>(void)> deltav1_obs = std::bind(&BlackScholesNetwork::get_delta_v1, &bsn);
    std::function<std::vector<double>(void)> deltav2_obs = std::bind(&ER_Network::delta_v2, this);

    // usage: register std::function with no parameters and boost::accumulator compatible return value. 2nd,... parameters are used to construct accumulator
    //S.register_observer(rs_obs, 2*N);
    S.register_observer(assets_obs, "Assets", N);
    S.register_observer(sol_obs, "Solvent", N);
    S.register_observer(valuation_obs, "Valuation", N);
    S.register_observer(std::bind(&ER_Network::sumM, this), "Sum over M", 1);
    //std::function<std::vector<double>(void)> out_obs = std::bind(&N2_network::test_out, this);
    //S.register_observer(out_obs, 1);

    S.register_observer(deltav1_obs, "Delta using Jacobians", 2 * N * N);
    S.register_observer(deltav2_obs, "Delta using Log", 2 * N * N);
    LOG(INFO) << "Running Valuation for N = " << N;

    S.draw_samples(f_run, f_dist, 3000);
    LOG(INFO) << std::endl << "Means: ";

    auto res = S.extract(MCUtil::StatType::MEAN);
    for (auto el : res) {
        std::cout << el.first << ": " << std::endl;
        Eigen::MatrixXd m;
        if (el.second.size() > N) {
            m = Eigen::MatrixXd::Map(&el.second[0], 2 * N, N);
            std::cout << m;
        } else {
            for (auto eli : el.second)
                std::cout << eli << ", ";
        }
        std::cout << std::endl << std::endl;
    }
    LOG(INFO) << std::endl << "Vars: ";
    auto res_var = S.extract(MCUtil::StatType::VARIANCE);
    for (auto el : res_var) {
        std::cout << el.first << ": " << std::endl;
        Eigen::MatrixXd m;
        if (el.second.size() > N) {
            m = Eigen::MatrixXd::Map(&el.second[0], 2 * N, N);
            std::cout << m << ", ";
        } else {
            for (auto eli : el.second)
                std::cout << eli << ", ";
        }
        std::cout << std::endl << std::endl;
    }
}

std::vector<double> ER_Network::draw_from_dist() {
    Eigen::VectorXd S_log(N);             // log of lognormal distribution exogenous assets, without a_0
    for (unsigned int d = 0; d < N; d++) {
        Z(d) = Z_dist(gen_z);
        S_log(d) = var_h(d) + std::sqrt(T) * Z(d);
    }
    std::vector<double> res;
    res.resize(S_log.size());
    Eigen::VectorXd::Map(&res[0], S_log.size()) = S_log.array().exp();
    return res;
}

std::vector<double> ER_Network::delta_v2() {
    //delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
    std::vector<double> res(2 * N * N);
    Eigen::VectorXd ln_fac = (itSigma * Z).array() / bsn.get_S0().array();
    Eigen::MatrixXd m = std::exp(-r * T) * (ln_fac * bsn.get_rs_eigen().transpose()).transpose();
    Eigen::MatrixXd::Map(&res[0], m.rows(), m.cols()) = m;
    return res;
}

void ER_Network::set_M_ER(const double p, const double val, char which_to_set)
{
    EXPECT_GT(val, 0) << "val is not a probability";
    EXPECT_LT(val, 1) << "val is not a probability";
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
    auto col_sum = M.colwise().sum();
    auto row_sum = M.rowwise().sum();
    double max = std::max(col_sum.maxCoeff(), row_sum.maxCoeff());
    M = (val/max)*M;
    bsn.set_M(M);
}
