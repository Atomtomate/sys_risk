/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#include "NetwSim.hpp"
void NetwSim::reset_network() {
    // ===== Reset Sampler =====
    results.clear();

    if(bsn != nullptr)
        delete bsn;
    LOG(TRACE) << "Generating Model";
    LOG(TRACE) << "creating new Black Scholes Network";

    bsn = new BlackScholesNetwork(S0, debt, sigma, T, r);
    LOG(TRACE) << "Initializing random connectivity matrix";
}

void NetwSim::init_2D_network(BSParameters& bs_params, const double vs01, const double vs10, const double vr01, const double vr10)
{
    N = 2;
    init_network(2, 0, 0, 1, bs_params.T, bs_params.r, bs_params.S0, bs_params.sigma, bs_params.default_prob_scale, NetworkType::Fixed2D);
    init_2DFixed_BS(vs01, vs10, vr01, vr10);
}

void NetwSim::init_network(const int N_, const double p_, const double val_, const int which_to_set, const double T_,\
    const double r_, const double S0_, const double sdev, const double default_prob_scale_, const NetworkType net_t_)
{
    // ========================= Initialization of temporary variables ========================
    T = T_; r = r_; N = N_; p = p_; val = val_; S0scalar = S0_; setM = which_to_set; default_prob_scale = default_prob_scale_; net_t = net_t_;
    Z.resize(N); iSigma.resize(N, N); var_h.resize(N); S0.resize(N); debt.resize(N); sigma.resize(N);

    // ========================= Preparation of Log Normal Dist. ==============================
    sigma = Eigen::VectorXd::Constant(N,sdev*sdev);
    iSigma = (sigma.asDiagonal()).inverse();
    var_h = T * r - T * sigma.array()/ 2.;

    // ============================== Importance Sampling Setup ===============================
    //Eigen::MatrixXd unitMatrix = Eigen::MatrixXd::Identity(N,N);
    //Eigen::VectorXd zeroVec = Eigen::VectorXd::Zero(N);
    //mvndist = Multivariate_Normal_Dist(unitMatrix, zeroVec);
    //t_dist = Student_t_dist(unitMatrix*4.0, zeroVec, deg_of_freedom);
    //chi_dist = trng::chi_square_dist(deg_of_freedom);

    // =============== creating correlated normal distribution from Eigen Sigma ===============
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


    // ========================= Generating Black Scholes Network =============================
    S0 = Eigen::VectorXd::Constant(N, S0scalar);
    //debt = (S0.array())*Eigen::VectorXd::Constant(N, default_prob_scale/(1.0-val)).array();
    debt = Eigen::VectorXd::Constant(N, 1.0*default_prob_scale);

    reset_network();
}



ResultType NetwSim::run_valuation(const long N_Samples, const long N_networks, const bool fixed_M) {

    std::map<int, std::unique_ptr<MCUtil::Sampler<AccType> > > SamplerList;
    Num_Samples = N_Samples;
    Num_Networks = N_networks;
    const auto f_dist_lambda = [this]() -> Eigen::MatrixXd { return this->draw_from_dist(); };
    std::function<Eigen::MatrixXd (void)> f_dist(std::cref(f_dist_lambda));
    //auto f_weights = [this]() -> double { return this->get_weights(); };
    const auto f_run_lambda =  [this](const Eigen::Ref<const Eigen::MatrixXd>& x) -> void {
        this->run(transformZ(x));
        //this->bsn->get_scalar_allGreeks(this->Z);
    };
    std::function<void (Eigen::MatrixXd)> f_run(std::cref(f_run_lambda));
    int networks = 0;
    const int conn = static_cast<int>(p*val);

    std::cout << "Running Valuation for N = " << N <<  ", p =" << p << ", sum_j M_ij = " << val  << "\n";
    std::cout << "Preparing to run"<< std::flush ;

    io_deg_dist = Eigen::MatrixXd::Zero(2,2*N);
    avg_rc_sums =  Eigen::MatrixXd::Zero(2,2*N);


    for(int jj = 0; jj < N_networks; jj++) // N_networks
    {
        std::cout << "\r  ---> " << 100.0*static_cast<double>(jj)/N_networks << "% of runs finished" <<std::flush;
        try{
            if(net_t == NetworkType::ER)
                init_BS(Utils::gen_sinkhorn);
            else if( net_t == NetworkType::Fixed2D)
            {
                // nothing to do -- fixed
            }
            else if( net_t == NetworkType::STAR)
                init_BS(Utils::gen_star);
            else if( net_t == NetworkType::RING)
                init_BS(Utils::gen_ring);
            else if( net_t == NetworkType::ER_SCALED)
                init_BS(Utils::gen_ring);
            else if( net_t == NetworkType::UNIFORM)
                init_BS(Utils::gen_uniform);
            int degree = (int)std::round(COARSE_CONN*(avg_io_deg.first + avg_io_deg.second)/2.);
            auto it = SamplerList.find(degree);
            if(it == SamplerList.end())
            {
                //LOG(ERROR) << "creating sampler for k = " << degree;
                std::unique_ptr<MCUtil::Sampler<AccType> > S = std::unique_ptr<MCUtil::Sampler<AccType> >(new MCUtil::Sampler<AccType>(N, bsn));
                if(S == nullptr) LOG(ERROR) << "Sampler could not be created";
                f_run(f_dist());
                S->draw_samples(f_run, f_dist, N_Samples);
                if ( !SamplerList.insert(std::make_pair(degree,std::move(S))).second ) {
                    LOG(ERROR) << "Sampler for degree=" << degree << "already found!";
                }
            }
            else {
                it->second->draw_samples(f_run, f_dist, N_Samples);
            }
            networks += 1;
        }
        catch (const std::runtime_error& e)
        {
            LOG(ERROR) << "Skipping uninitialized network for N = " << N << ", p=" << p << ", val=" << val;
            initialized = 0;
        }
    }
    std::cout << "\r" << std::endl;

    initialized = false;
    for (auto it = SamplerList.begin(); it != SamplerList.end(); ++it)
    {
        auto res = it->second->extract();
        if ( !results.insert( std::make_pair( it->first, res ) ).second ) {
            LOG(ERROR) << "result for <k>=" << it->first << "already found!";
        }
    }
    //SamplerList.clear();
    return results;
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
    return Z;
}

const Eigen::MatrixXd NetwSim::transformZ(const Eigen::Ref<const Eigen::MatrixXd>& Z) const
{
    //Z = sw*2.0*Z;
    Eigen::VectorXd S_log = var_h.array() + (std::sqrt(T) * sigma.array().sqrt()).array() * Z.array();
    //set_weight();
    return S_log.array().exp();
}
