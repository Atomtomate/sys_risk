/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "NetwSim.hpp"
void NetwSim::test_init_network() {
    // ===== Reset Sampler =====
    auto it = SamplerList.begin();
    while(it != SamplerList.end())
    {
        if(it->second != nullptr)
            delete it->second;
        it++;
    }
    SamplerList.clear();
    results.clear();

    if(bsn != nullptr)
        delete bsn;
    LOG(TRACE) << "Generating Model";
    LOG(TRACE) << "creating new Black Scholes Network";

    bsn = new BlackScholesNetwork(S0, debt, sigma, T, r);
    LOG(TRACE) << "Initializing random connectivity matrix";
}

void NetwSim::test_init_network(const long N_, const double p_, const double val_, const int which_to_set, const double T_, const double r_, const double S0_, const double sdev, const double default_prob_scale_) {

    // ===== Initialization of temporary variables =====
    T = T_; r = r_; N = N_; p = p_; val = val_; S0scalar = S0_; setM = which_to_set; default_prob_scale = default_prob_scale_;
    Z.resize(N);
    iSigma.resize(N, N);
    var_h.resize(N);
    S0.resize(N);
    debt.resize(N);
    sigma.resize(N);
    io_deg_dist.resize(2, N);
    avg_rc_sums.resize(2,N);
    io_deg_dist = Eigen::MatrixXd::Zero(2, 2*N);
    avg_rc_sums = Eigen::MatrixXd::Zero(2, 2*N);

    // ===== Preparation of Log Normal Dist. =====
    sigma = Eigen::VectorXd::Constant(N,sdev*sdev);
    iSigma = (sigma.asDiagonal()).inverse();
    var_h = T * r - T * sigma.array()/ 2.;

    //Eigen::MatrixXd unitMatrix = Eigen::MatrixXd::Identity(N,N);
    //Eigen::VectorXd zeroVec = Eigen::VectorXd::Zero(N);
    //mvndist = Multivariate_Normal_Dist(unitMatrix, zeroVec);
    //t_dist = Student_t_dist(unitMatrix*4.0, zeroVec, deg_of_freedom);
    //chi_dist = trng::chi_square_dist(deg_of_freedom);

    // ===== creating correlated normal distribution from Eigen Sigma =====
    //double *sigma_arr = new double[N * N];
    //double sigma2d_arr[N][N];
    //memcpy(sigma2d_arr[0], sigma_arr, N*N*sizeof(double));
    //delete[] sigma_arr;
    double *unity_arr = new double[N * N];
    double unity2d_arr[N][N];
    Eigen::MatrixXd::Map(unity_arr, N, N) = Eigen::MatrixXd::Identity(N, N);
    memcpy(unity2d_arr[0], unity_arr, N*N*sizeof(double));

    //Z_dist = trng::correlated_normal_dist<>(&sigma2d_arr[0][0], &sigma2d_arr[N - 1][N - 1] + 1);
    Z_dist = trng::correlated_normal_dist<>(&unity2d_arr[0][0], &unity2d_arr[N - 1][N - 1] + 1);


    delete[] unity_arr;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, std::numeric_limits<int>::max());
    gen_z.seed(dis(gen));
    gen_chi.seed(dis(gen));


    // ===== Generating Black Scholes Network =====
    S0 = Eigen::VectorXd::Constant(N, S0scalar);
    //debt = (S0.array())*Eigen::VectorXd::Constant(N, default_prob_scale/(1.0-val)).array();
    debt = Eigen::VectorXd::Constant(N, 1.0*default_prob_scale);

    test_init_network();
}



ResultType NetwSim::run_valuation(const long N_Samples, const long N_networks, const bool fixed_M) {
    test_init_network();

    const auto f_dist = [this]() -> Eigen::MatrixXd { return this->draw_from_dist(); };
    //auto f_weights = [this]() -> double { return this->get_weights(); };
    const auto f_run =  [this](const Eigen::Ref<const Eigen::MatrixXd>& x) -> BlackScholesNetwork* { this->run(x); return bsn; };
    int networks = 0;
    const int conn = static_cast<int>(p*val);

    std::cout << "Running Valuation for N = " << N <<  ", p =" << p << ", sum_j M_ij = " << val  << "\n";
    std::cout << "Preparing to run"<< std::flush ;

    MCUtil::Sampler<AccType>* S = new MCUtil::Sampler<AccType>();
    register_observers<AccType>(S);

    for(int jj = 0; jj < N_networks; jj++)
    {
        std::cout << "\r  ---> " << 100.0*static_cast<double>(jj)/N_networks << "% of runs finished" <<std::flush;

        init_BS(Utils::gen_sinkhorn);
        f_run(f_dist());
        bsn->get_scalar_allGreeks(Z);
        S->draw_samples(f_run, f_dist, N_Samples);

        /*
        try{
            //init_BS(Utils::gen_configuration_model);
            init_BS(Utils::gen_sinkhorn);
            //init_BS(Utils::fixed_2d);
#if USE_ACTUAL_CONN
            int degree = (int)std::round(5.*(avg_io_deg.first + avg_io_deg.second)/2.);
            auto it = SamplerList.find(degree);
            if(it == SamplerList.end())
            {
                MCUtil::Sampler<AccType>* S = new MCUtil::Sampler<AccType>();
                register_observers<AccType>(S);
                S->draw_samples(f_run, f_dist, N_Samples);
                SamplerList.insert(std::pair(degree,S));
                it = SamplerList.find(degree);
            }
            else {
                (*it).second->draw_samples(f_run, f_dist, N_Samples);
            }
#endif
            networks += 1;
        }
        catch (const std::runtime_error& e)
        {
            LOG(ERROR) << "Skipping uninitialized network for N = " << N << ", p=" << p << ", val=" << val;
            initialized = 0;
        }
         */
    }
    std::cout << "\r" << std::endl;
    //io_deg_dist /= networks;
    //avg_rc_sums /= networks;


    //auto io_deg_obs_lambda = [this]() -> Eigen::MatrixXd { return this->io_deg_dist; };
    //std::function<const Eigen::MatrixXd(void)> io_deg_obs(std::cref(io_deg_obs_lambda));
    //S->register_observer(io_deg_obs, io_deg_str, 2 , N);

    initialized = false;
    ResultType result;
    for(auto el : SamplerList)
    {
        result.insert( std::pair(el.first, result_object<AccType>(el.first, el.second)) );
    }
    return result;
}




const Eigen::MatrixXd NetwSim::delta_v2() {
    //delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
    Eigen::VectorXd ln_fac = (iSigma * Z/T).array() / (bsn->get_S0()).array();
    Eigen::MatrixXd m = std::exp(-r * T) * (ln_fac * (bsn->get_rs()).transpose()).transpose();
    //Eigen::MatrixXd::Map(&res[0], m.rows(), m.cols()) = m;
    return m;
}

void NetwSim::set_weight()
{
    double weight = 1.0;//std::exp(mvndist.logpdf(Z) - t_dist.logpdf(Z));
    //dbg_weights.push_back(weight);
    last_weight = weight;
}


const Eigen::MatrixXd NetwSim::draw_from_dist()
{
    //double sw = std::sqrt(deg_of_freedom/chi_dist(gen_chi));
    for (int d = 0; d < N; d++) {
        Z(d) = Z_dist(gen_z);
    }
    //Z = sw*2.0*Z;

    Eigen::VectorXd S_log = var_h.array() + (std::sqrt(T) * sigma.array().sqrt()).array() * Z.array();
    //set_weight();
    return S_log.array().exp();
}

