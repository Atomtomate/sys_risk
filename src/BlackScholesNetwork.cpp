/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#include "BlackScholesNetwork.hpp"


BlackScholesNetwork::BlackScholesNetwork(const double T, const double r):
        T(T), r(r), exprt(std::exp(-r * T))
{
    initialized = false;
    //EXPECT_EQ(M.cols(), 2 * M.rows()) << "Dimensions for cross holding matrix invalid!";
}


BlackScholesNetwork::BlackScholesNetwork(const Eigen::Ref<Mat>& M_, const Eigen::Ref<Vec>& S0, const Eigen::Ref<Vec>& assets, const Eigen::Ref<Vec>& debt, const double T, const double r):
        N(M_.rows()), S0(S0), St(assets), debt(debt), T(T), r(r), exprt(std::exp(-r * T))
{
    initialized = true;
    x = Eigen::VectorXd::Zero(2*N);
#ifdef USE_SPARSE_INTERNAL
    Id.resize(2*N, 2*N);
    Id.setIdentity();
    //Jrs.resize(2*N,2*N);
    J_a.resize(2*N, N);
    M = M_.sparseView();
    M.makeCompressed();
#else
    M = M_;
    lu = Eigen::PartialPivLU<Eigen::MatrixXd>(2*N);
    //Jrs = Eigen::MatrixXd::Zero(2*N, 2*N);
    J_a = Eigen::MatrixXd::Zero(2*N, N);
    x = Eigen::VectorXd::Zero(2*N);
#endif
};


void BlackScholesNetwork::set_solvent()
{
    solvent.resize(N);
    for(int i = 0; i < N; i++) {
        solvent(i) = 1*(x(i)+x(i+N) >= debt(i));
    }
}


const Eigen::MatrixXd BlackScholesNetwork::run_valuation(unsigned int iterations)
{
    if(!initialized) throw std::logic_error("attempting to solve uninitialized model!");
    x.resize(2*N);
    auto x_old = x;

    double dist = 99.;
    Eigen::MatrixXd a = S0.array()*St.array();
    while(dist > 1.0e-12)
    {
        auto tmp = (a + M*x);
        x_old = x;
        x.head(N) = (tmp - debt).array().max(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        dist = (x_old - x).norm(); //.lpNorm<Eigen::Infinity>();//
    }
    set_solvent();
    return x;
}


//@TODO: check if return Eigen::Refwould be better here
const Eigen::VectorXd BlackScholesNetwork::get_assets()
{
    return (S0.array()*St.array());
}


const Eigen::MatrixXd BlackScholesNetwork::get_delta_v1() {
    //TIMED_FUNC(timerObj);
#ifdef USE_SPARSE_INTERNAL
    J_a.setZero();
    for(int i = 0; i < N; i++)
    {
        J_a.insert(i,i) = solvent(i);
        J_a.insert(i+N,i) = (1.-solvent(i));
    }
    //PERFORMANCE_CHECKPOINT(timerObj);
    //Jrs = Z*M;
    J_a.makeCompressed();
    //PERFORMANCE_CHECKPOINT(timerObj);
    lu.compute(Id - J_a*M);
    auto Jrs = lu.solve(Id);
    //PERFORMANCE_CHECKPOINT(timerObj);
    Eigen::MatrixXd res_eigen = exprt*(Jrs*J_a)*St.asDiagonal();
    //PERFORMANCE_CHECKPOINT(timerObj);
#else
    auto msol = 1-solvent.array();
        J_a.diagonal(0) = solvent;
        J_a.diagonal(-N) = msol;
        //Jrs.topRows(N) = M.array().colwise() * solvent.array();
        //Jrs.bottomRows(N) = M.array().colwise() * msol;
        Jrs = J_a*M;
        if(Jrs.isIdentity(0.001)) return Eigen::MatrixXd::Zero(2*N, N);
        auto res_eigen =  exprt*(lu.compute(Eigen::MatrixXd::Identity(2*N, 2*N) - Jrs).inverse()*J_a)*(St.asDiagonal());
#endif
    return res_eigen;
    //Eigen::MatrixXd::Map(&res[0], res_eigen.rows(), res_eigen.cols()) = res_eigen;
    //return res;
}




void BlackScholesNetwork::debug_print()
{
    LOG(DEBUG) << "DEBUG PRINT BLACK SCHOLES NETWORK";
    Eigen::IOFormat CleanFmt(3, 0, " ", "\n", "[", "]");
#ifdef USE_SPARSE_INTERNAL
    //LOG(DEBUG) << "initialized: " << initialized << ", N = " << N <<", T = " << T << ", r = " << r << ", exprt = " << exprt << "M: \n" << M.format(CleanFmt);
#else
    LOG(DEBUG) << "initialized: " << initialized << ", N = " << N <<", T = " << T << ", r = " << r << ", exprt = " << exprt;
    LOG(DEBUG) << "M: \n" << Eigen::MatrixXd(M).format(CleanFmt);
#endif
    LOG(DEBUG) << "\nS0 \n" << S0;
    LOG(DEBUG) << "\nSt: \n"  << St;
    LOG(DEBUG) << "\ndebt: \n" << debt;
}


