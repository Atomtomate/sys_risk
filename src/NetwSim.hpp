/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#ifndef VALUATION_NETWORK_SIM_HPP
#define VALUATION_NETWORK_SIM_HPP

#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <limits>
#include <random>
#include <vector>

#include "trng/chi_square_dist.hpp"

#include "Utils.hpp"

#ifdef USE_MPI

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#endif

#include "StudentT.hpp"
#include "MVarNormal.hpp"
#include "Sampler.hpp"
#include "StatAcc.hpp"
#include "BlackScholesNetwork.hpp"
#include "RndGraphGen.hpp"

struct Parameters
{
    long N;                 // M.rows()
    int set_s_d_both;       //
    double p;               // P(M_ij = 1)
    double val;             // sum_j M_ij = sum_i M_ij
    //@TODO: finish, add log-norm params


};

constexpr int deg_of_freedom = 8;
//const std::string io_deg_str("In/Out degree distribution");

class NetwSim {
    friend class Py_ER_Net;
private:

#ifdef USE_MPI
    const boost::mpi::communicator local;
    const boost::mpi::communicator world;
    const bool isGenerator;
#endif
    trng::yarn2 gen_u;
    trng::uniform01_dist<> u_dist;
    Student_t_dist t_dist;
    Multivariate_Normal_Dist mvndist;
    //std::vector<double> dbg_weights;
    double last_weight;

    long N;
    bool initialized;
    double T;              // maturity
    double r;              // interest
    double p;
    double val;
    double S0scalar;
    double sigmaScalar;
    double default_prob_scale;
    int setM;
    const double tmp[2][2] = {{1, 0},
                              {0, 1}};
    BlackScholesNetwork* bsn;
    Eigen::MatrixXd iSigma;
    Eigen::VectorXd sigma;
    Eigen::VectorXd Z;                 // Multivariate normal, used to generate lognormal assets
    Eigen::VectorXd var_h;
    std::map<int, MCUtil::Sampler<Eigen::MatrixXd>*> SamplerList;
    Eigen::VectorXd S0;
    Eigen::VectorXd debt;
    Eigen::MatrixXd io_deg_dist;
    Eigen::MatrixXd avg_rc_sums;
    std::pair<double, double> avg_io_deg;

    std::map<int, std::unordered_map<std::string, Eigen::MatrixXd> > results;
    double connectivity;


    // last result, returned by observe
    void test_init_network();

    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd* M);

    template <typename F>
    void init_BS(F gen_function) {
        if (val < 0 || val >= 1) throw std::logic_error("Row sum is not in [0,1)");
        if (p < 0 || p > 1) throw std::logic_error("p is not a probability");
        connectivity = N * p;
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, 2 * N);
        gen_function(&M, gen_u, p, val, setM);
        //Utils::gen_fixed_degree(&M, gen_u, p, val, which_to_set);
        io_deg_dist += Utils::in_out_degree(&M);
        avg_io_deg = Utils::avg_io_deg(&M);
        avg_rc_sums += Utils::avg_row_col_sums(&M);
        bsn->re_init(M, S0, debt);
    }


public:
    /*!
     * @brief               (re-)initializes network to given parameters
     * @param N             Size of network
     * @param p             Probability of cross holding
     * @param val           total value in/being held by other firms
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     */
    void test_init_network(const long N_, const double p_, const double val_, const int which_to_set, const double T_, const double r_, const double S0_, const double sigma_, const double default_prob_scale_);

    virtual ~NetwSim(){
        if(bsn != nullptr)
            delete bsn;
        /*auto it = SamplerList.begin();
        while(it != SamplerList.end())
        {
            if(it->second != nullptr)
                delete it->second;
        }
        SamplerList.clear();
         */
    }

    /*!
     * @brief               Constructs the Black Scholes Model using random cross holdings.
     * @param local         local MPI communicator (between producers/consumers only)
     * @param world         global MPI communicator
     * @param isGenerator   Flag for generator/consumer ranks
     */
#ifdef USE_MPI
    ER_Network(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator):
            local(local), world(world), isGenerator(isGenerator), Z_dist(&tmp[0][0], &tmp[1][1]), chi_dist(deg_of_freedom), t_dist(deg_of_freedom)
#else
    NetwSim():
            Z_dist(&tmp[0][0], &tmp[1][1]), chi_dist(deg_of_freedom), t_dist(deg_of_freedom), initialized(false)
#endif
    {
        bsn = nullptr;
        iSigma = Eigen::MatrixXd::Zero(1,1);
        Z = Eigen::VectorXd::Zero(1,1);
        var_h = Eigen::VectorXd::Zero(1,1);
    }

    /*!
     * @brief               Constructs the Black Scholes Model using random cross holdings.
     * @param local         local MPI communicator (between producers/consumers only)
     * @param world         global MPI communicator
     * @param isGenerator   Flag for generator/consumer ranks
     * @param N             Size of network
     * @param p             Probability of connection between firms
     * @param val
     * @param which_to_set  Flag to disable connections between parts of the network. Can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
     * @param T             maturity
     * @param r             interest rate
     */
#ifdef USE_MPI
    NetwSim(const boost::mpi::communicator local, const boost::mpi::communicator world, const bool isGenerator,
               long N, double p, double val, int which_to_set, const double T, const double r, const double S0) :
            local(local), world(world), isGenerator(isGenerator),
#else
    NetwSim(long N_, double p_, double val, int which_to_set, const double T_, const double r_, const double S0_, const double sigma_, const double default_scale_) :
#endif
            val(val), T(T_), r(r_), S0scalar(S0_), sigmaScalar(sigma_), default_prob_scale(default_scale_)\
        , Z_dist(&tmp[0][0], &tmp[1][1]), chi_dist(deg_of_freedom), t_dist(deg_of_freedom)
    {
        gen_u.seed();
        bsn = nullptr;
        test_init_network(N_, p_, val, which_to_set, T_, r_, S0_, sigma_, default_scale_);
    }



    /*!
     * @brief       Runs a series of example simulations
     * @param N_in  Size of network
     */
    std::map<int, std::unordered_map<std::string, Eigen::MatrixXd>> run_valuation(const long N_Samples = 2000,
                                                                   const long N_networks = 100, const bool fix_degree = false);

    /*!
     * @brief   Draws a random number from a multivariate lognormal distribution
     * @return  Random sample from a multivariate lognormal distribution
     */
    const Eigen::MatrixXd draw_from_dist();

    double get_weight();

    /*!
     * @brief       Runs a single simulation of the Black Scholes model to find the fix point valuation.
     * @param St_in Initial asset value
     * @return      Valuation of firms at maturity T
     */
    auto run(const Eigen::Ref<const Eigen::VectorXd>& St_in)//Eigen::VectorXd St)
    {
        //Eigen::VectorXd St = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(St_in.data(), St_in.size());
        bsn->set_St(St_in);
        bsn->run_valuation(1000);
    }

    /*!
     * @brief   Compute \f$\Delta\f$ using the covariance matrix of the normal distribution
     * @return  \f$\Delta\f$
     */
    const Eigen::MatrixXd delta_v2();

    /*!
     * @brief   Computes the sum over all elements of the cross holdings matrix
     * @return  \f$\sum_{ij} M_{ij}\f$
     */
    std::vector<double> sumM() {
        std::vector<double> res{(bsn->get_M()).sum()};
        return res;
    }

    Eigen::MatrixXd get_M()
    {
        return bsn->get_M();
    }

    auto test_out()
    {
        auto v_o = bsn->get_valuation();
        auto s_o = bsn->get_solvent();
        std::cout << "output after sample: " << std::endl;
        LOG(INFO) << "Valuation: \n" << v_o;
        LOG(INFO) << "solvent: \n" << s_o;
        LOG(INFO) << "St: \n" << bsn->get_assets();
        LOG(INFO) << "debt: \n" << bsn->get_debt();
        LOG(INFO) << "M: \n" << bsn->get_M();
        std::cout << "------" << std::endl;
        Eigen::MatrixXd out = Eigen::MatrixXd::Constant(1,1,0);
        return out;
    }


private:

    trng::yarn2 gen_z;
    trng::yarn2 gen_chi;
    trng::chi_square_dist<double> chi_dist;
    trng::correlated_normal_dist<> Z_dist;

    template<typename T>
    std::unordered_map<std::string, Eigen::MatrixXd> result_object(const int k, MCUtil::Sampler<T>* S, const long N_Samples, const long N_networks)
    {

        const std::string count_str("#Samples");
        const std::string rs_str("RS");
        const std::string M_str("M");
        const std::string assets_str("Assets");
        const std::string solvent_str("Solvent");
        const std::string val_str("Valuation");
        const std::string delta1_str("Delta using Jacobians");
        const std::string delta2_str("Delta using Log");
        const std::string rho_str("Rho");
        const std::string theta_str("Theta");
        const std::string vega_str("Vega");
        const std::string pi_str("Pi");
        const std::string io_deg_str("In/Out degree distribution");
        const std::string io_weight_str("In/Out weight distribution");

        Eigen::MatrixXd count;
        Eigen::MatrixXd mean_delta_jac;
        Eigen::MatrixXd mean_delta_log;
        Eigen::MatrixXd mean_rho;
        Eigen::MatrixXd mean_theta;
        Eigen::MatrixXd mean_vega;
        Eigen::MatrixXd mean_assets;
        Eigen::MatrixXd mean_rs;
        Eigen::MatrixXd mean_M;
        Eigen::MatrixXd mean_solvent;
        Eigen::MatrixXd mean_valuation;
        Eigen::MatrixXd mean_io_deg_dist;
        Eigen::MatrixXd mean_io_weight_dist;
        Eigen::MatrixXd mean_pi;
        Eigen::MatrixXd var_delta_jac;
        Eigen::MatrixXd var_delta_log;
        Eigen::MatrixXd var_rho;
        Eigen::MatrixXd var_theta;
        Eigen::MatrixXd var_vega;
        Eigen::MatrixXd var_assets;
        Eigen::MatrixXd var_rs;
        Eigen::MatrixXd var_M;
        Eigen::MatrixXd var_solvent;
        Eigen::MatrixXd var_valuation;
        Eigen::MatrixXd var_io_deg_dist;
        Eigen::MatrixXd var_io_weight_dist;
        Eigen::MatrixXd var_pi;

        std::unordered_map<std::string, Eigen::MatrixXd> res;
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
            else if(el.first.compare(rho_str) == 0){ mean_rho = el.second; res[rho_str] = el.second;}
            else if(el.first.compare(theta_str) == 0){ mean_theta = el.second; res[theta_str] = el.second;}
            else if(el.first.compare(vega_str) == 0){ mean_vega = el.second; res[vega_str] = el.second;}
            else if(el.first.compare(pi_str) == 0){ mean_pi = el.second; res[pi_str] = el.second;}
            else if(el.first.compare(io_deg_str) == 0){ mean_io_deg_dist = el.second; res[io_deg_str] = el.second;}
            else if(el.first.compare(io_weight_str) == 0){ mean_io_weight_dist = el.second; res[io_weight_str] = el.second;}
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
            else if(el.first.compare(rho_str) == 0){ var_rho = el.second; res["Variance "+rho_str] = el.second;}
            else if(el.first.compare(theta_str) == 0){ var_theta = el.second; res["Variance "+theta_str] = el.second;}
            else if(el.first.compare(vega_str) == 0){ var_vega = el.second; res["Variance "+vega_str] = el.second;}
            else if(el.first.compare(pi_str) == 0){ var_pi = el.second; res["Variance "+pi_str] = el.second;}
            else if(el.first.compare(io_deg_str) == 0){ var_io_deg_dist = el.second; res["Variance" + io_deg_str] = el.second;}
            else if(el.first.compare(io_weight_str) == 0){ var_io_weight_dist = el.second; res["Variance" + io_weight_str] = el.second;}
            else LOG(WARNING) << "result " << el.first << ", not saved";
        }
        results.insert(std::pair(k, res));
        return res;
    }

    template<typename T>
    void register_observers(MCUtil::Sampler<T>* S)
    {

const std::string count_str("#Samples");
const std::string rs_str("RS");
const std::string M_str("M");
const std::string assets_str("Assets");
const std::string solvent_str("Solvent");
const std::string val_str("Valuation");
const std::string delta1_str("Delta using Jacobians");
const std::string delta2_str("Delta using Log");
const std::string rho_str("Rho");
const std::string theta_str("Theta");
const std::string vega_str("Vega");
const std::string pi_str("Pi");


        // ===== Defining observables =====
        auto asset_obs_lambda = [this]() -> Eigen::MatrixXd { return  bsn->get_assets(); };
        auto rs_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_rs(); };
        auto M_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_M(); };
        auto sol_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_solvent(); };
        auto delta_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_delta_v1();};
        auto rho_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_rho();};
        auto theta_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_theta(Z);};
        auto vega_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_vega(Z);}; //LOG(ERROR) << "a"; Eigen::MatrixXd t = Eigen::MatrixXd::Zero(2,2);
        auto pi_obs_lambda = [this]() -> Eigen::MatrixXd { return bsn->get_pi();};
        std::function<const Eigen::MatrixXd(void)> assets_obs(std::ref(asset_obs_lambda));
        S->register_observer(assets_obs, assets_str, N, 1);
        std::function<const Eigen::MatrixXd(void)> rs_obs(std::cref(rs_obs_lambda));
        S->register_observer(rs_obs, rs_str, 2*N, 1);
        std::function<const Eigen::MatrixXd(void)> M_obs(std::cref(M_obs_lambda));
        S->register_observer(M_obs, M_str, N, 2*N);
        std::function<const Eigen::MatrixXd(void)> sol_obs(std::cref(sol_obs_lambda));
        S->register_observer(sol_obs, solvent_str, N, 1);
        std::function<const Eigen::MatrixXd(void)> deltav1_obs(std::cref(delta_obs_lambda));
        S->register_observer(deltav1_obs, delta1_str, 2 * N , N);
        std::function<const Eigen::MatrixXd(void)> rho_obs(std::cref(rho_obs_lambda));
        S->register_observer(rho_obs, rho_str, 2*N , 1);
        std::function<const Eigen::MatrixXd(void)> theta_obs(std::cref(theta_obs_lambda));
        S->register_observer(theta_obs, theta_str, 2*N , 1);
        std::function<const Eigen::MatrixXd(void)> vega_obs(std::cref(vega_obs_lambda));
        S->register_observer(vega_obs, vega_str, 2*N , N);
        std::function<const Eigen::MatrixXd(void)> pi_obs(std::cref(pi_obs_lambda));
        S->register_observer(pi_obs, pi_str, N , 1);

        //std::function<const Eigen::MatrixXd(void)> deltav2_obs   = [this]() -> Eigen::MatrixXd { return this->delta_v2();};
        //S->register_observer(deltav2_obs, delta2_str, 2 * N , N);
        //std::function<const Eigen::MatrixXd(void)> out_obs =  [this]() -> Eigen::MatrixXd { return this->test_out();};
        //S->register_observer(out_obs, "Debug Out" ,1, 1);

        //std::function<const Eigen::MatrixXd(void)> valuation_obs = [this]() -> Eigen::MatrixXd { return bsn->get_valuation(); };
        //S->register_observer(valuation_obs, val_str, N, 1);

    }

public:
    Eigen::MatrixXd get_io_deg_dist() const
    {
        return io_deg_dist;
    }

    Eigen::MatrixXd get_avg_row_col_sums() const
    {
        return avg_rc_sums;
    }

    void set_weight();

    //std::vector<double> get_dbg_weights() const
    //{
    //    return dbg_weights;
    //}

};


#endif //VALUATION_NETWORK_SIM_HPP
