/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "NetwSim.hpp"
void NetwSim::test_init_network() {
    test_init_network(N, p, val, setM, T, r, default_prob_scale);
}

void NetwSim::test_init_network(const long N_, const double p_, const double val_, const int which_to_set, const double T_, const double r_, const double default_prob_scale_) {
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

    // ===== Initialization of temporary variables =====
    T = T_; r = r_; N = N_; p = p_; val = val_; setM = which_to_set; default_prob_scale = default_prob_scale_;
    Z.resize(N);
    itSigma.resize(N, N);
    var_h.resize(N);
    S0.resize(N);
    debt.resize(N);

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
    debt = ((var_h + S0).array() - T*r)*Eigen::VectorXd::Constant(N, default_prob_scale/(1.0-val)).array();
    if(bsn != nullptr)
        delete bsn;
    LOG(TRACE) << "Generating Model";
    LOG(TRACE) << "creating new Black Scholes Network";
    bsn = new BlackScholesNetwork(T, r, sigma);
    LOG(TRACE) << "Initializing random connectivity matrix";
    /*try{
        init_M(Utils::gen_configuration_model);
        //init_M(Utils::gen_sinkhorn);
        initialized = true;
    }
    catch (const std::runtime_error& e)
    {
        LOG(ERROR) << "unable to create ER Model graph for N=" << N << ", p=" << p << ", val="<< val << "\nReason: " << e.what();
        initialized = false;
    }*/
}



std::map<int, std::unordered_map<std::string, Eigen::MatrixXd>> NetwSim::run_valuation(const long N_Samples, const long N_networks, const bool fix_degree) {
    test_init_network();

    auto f_dist = [this]() -> Eigen::MatrixXd { return this->draw_from_dist(); };
    auto f_run =  [this](const Eigen::Ref<const Eigen::MatrixXd>& x) { this->run(x); };

    std::cout << "Running Valuation for N = " << N <<  ", p =" << p << ", sum_j M_ij = " << val  << "\n";
    std::cout << "Preparing to run"<< std::flush ;
    for(int jj = 0; jj < N_networks; jj++)
    {
        std::cout << "\r  ---> " << 100.0*static_cast<double>(jj)/N_networks << "% of runs finished" <<std::flush;
        try{
            init_M(Utils::gen_configuration_model);
            //init_M(Utils::gen_sinkhorn);

            int degree = (int)std::round(5.*(avg_io_deg.first + avg_io_deg.second)/2.);
            auto it = SamplerList.find(degree);
            if(it == SamplerList.end())
            {
                MCUtil::Sampler<Eigen::MatrixXd>* S = new MCUtil::Sampler<Eigen::MatrixXd>();
                register_observers<Eigen::MatrixXd>(S);
                //LOG(INFO) << "aaaa";
                //f_run(ttt);
                //LOG(INFO) << "bbbb";
                S->draw_samples(f_run, f_dist, N_Samples);
                SamplerList.insert(std::pair(degree,S));
                it = SamplerList.find(degree);
            }
            else
                (*it).second->draw_samples(f_run, f_dist, N_Samples);

        } catch (const std::runtime_error& e)
        {
            LOG(ERROR) << "Skipping uninitialized network for N = " << N << ", p=" << p << ", val=" << val;
            initialized = 0;
        }
    }
    std::cout << "\r" << std::endl;


    //auto io_deg_obs_lambda = [this]() -> Eigen::MatrixXd { return this->io_deg_dist; };
    //std::function<const Eigen::MatrixXd(void)> io_deg_obs(std::cref(io_deg_obs_lambda));
    //S->register_observer(io_deg_obs, io_deg_str, 2 , N);

    initialized = false;
    std::map<int, std::unordered_map<std::string, Eigen::MatrixXd> > result;
    for(auto el : SamplerList)
    {
        result.insert( std::pair(el.first, result_object<Eigen::MatrixXd>(el.first, el.second, N_Samples, N_networks)) );
    }
    return result;
}




const Eigen::MatrixXd NetwSim::delta_v2() {
    //delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
    Eigen::VectorXd ln_fac = (itSigma * Z).array() / (bsn->get_S0()).array();
    Eigen::MatrixXd m = std::exp(-r * T) * (ln_fac * (bsn->get_rs()).transpose()).transpose();
    //Eigen::MatrixXd::Map(&res[0], m.rows(), m.cols()) = m;
    return m;
}


const Eigen::MatrixXd NetwSim::draw_from_dist() {
    for (long d = 0; d < N; d++) {
        Z(d) = Z_dist(gen_z);
    }
    Eigen::VectorXd S_log = var_h + std::sqrt(T) * Z;
    return S_log.array().exp();
}

