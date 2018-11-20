/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#include "BlackScholesNetwork.hpp"


BlackScholesNetwork::BlackScholesNetwork(const Eigen::Ref<Vec>& S0_, const Eigen::Ref<Vec>& debt_, const Eigen::Ref<Vec>& sigma_, const double T_, const double r_):
        T(T_), r(r_), exprt(std::exp(-r_ * T_))
{
    if(T == 0.) LOG(ERROR) << "T == 0 found!";
    N = S0_.size();
    if(N == 0)
    {
        LOG(WARNING) << "Zero size covariance matrix encountered in BlackScholes Network constructor!";
    }
    S0.resize(N);
    debt.resize(N);
    sigma.resize(N);
    St_full.resize(N);
    S0 = S0_;
    debt = debt_;
    sigma = sigma_;
    GreekMat.resize(2*N, 2*N);
    //sigma_diag.resize(2*N,N);
    //sigma_diag.topRows(N) = Sigma;
    //sigma_diag.bottomRows(N) = Sigma;
    initialized = false;
    jacobian_set = false;
    //EXPECT_EQ(M.cols(), 2 * M.rows()) << "Dimensions for cross holding matrix invalid!";
}


BlackScholesNetwork::BlackScholesNetwork(const Eigen::Ref<Mat>& M_, const Eigen::Ref<Vec>& S0_, const Eigen::Ref<Vec>& assets_, const Eigen::Ref<Vec>& debt_, const Eigen::Ref<Vec>& sigma_, const double T, const double r):
        T(T), r(r), exprt(std::exp(-r * T))
{
    if(T == 0.) LOG(ERROR) << "T == 0 found!";
    N = M_.rows();
    initialized = true;
    jacobian_set = false;
    x.resize(2*N);
    x = Eigen::VectorXd::Zero(2*N);
    S0.resize(N);
    debt.resize(N);
    sigma.resize(N);
    St_full.resize(N);
    St.resize(N);
    GreekMat.resize(2*N, 2*N);
    S0 = S0_;
    debt = debt_;
    sigma = sigma_;
    St_full = S0.array()*St.array();
#if USE_SPARSE_INTERNAL
    Id.resize(2*N, 2*N);
    Id.setIdentity();
    M = M_.sparseView();
    M.makeCompressed();
#else
    M = M_;
    lu = Eigen::PartialPivLU<Eigen::MatrixXd>(2*N);
    Id.resize(2*N, 2*N);
    Id.setIdentity();
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
    //if(!initialized) throw std::logic_error("attempting to solve uninitialized model!");
    x.resize(2*N);
    auto x_old = x;
    double dist = 99.;
    while(dist > 1.0e-12)
    {
        auto tmp = (St_full + M*x);
        x_old = x;
        x.head(N) = (tmp - debt).array().max(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        dist = (x_old - x).norm(); //.lpNorm<Eigen::Infinity>();//
    }
    jacobian_set = false;
    set_jacobian();
    //debug_print();
    return x;
}


//@TODO: check if return Eigen::Refwould be better here
const Eigen::VectorXd BlackScholesNetwork::get_assets()
{
    return St_full;
}


void BlackScholesNetwork::set_jacobian() {
    set_solvent();
    //TIMED_FUNC(timerObj);
#if USE_SPARSE_INTERNAL
    Eigen::SparseMatrix<double, Eigen::ColMajor> J_rs;
    Eigen::SparseMatrix<double, Eigen::ColMajor> J_a;
    J_a.resize(2*N, N);
    J_rs.resize(2*N, 2*N);
    GreekMat.resize(2*N, N);
    J_a.setZero();
    for(int i = 0; i < N; i++)
    {
        J_a.insert(i,i) = solvent(i);
        J_a.insert(i+N,i) = (1.-solvent(i));
    }
    J_a.makeCompressed();
    lu.compute(Id - J_a*M);
    const auto inv = lu.solve(Id);
    GreekMat = inv * J_a;
#else
    auto msol = 1-solvent.array();
    Eigen::MatrixXd J_a = Eigen::MatrixXd::Zero(2*N, N);
    Eigen::MatrixXd J_rs(2*N, 2*N);
    GreekMat.resize(2*N, N);
    J_a.diagonal(0) = solvent;
    J_a.diagonal(-N) = msol;
    J_rs = J_a*M;
    if(J_rs.isIdentity(0.001))
        GreekMat = Eigen::MatrixXd::Zero(2*N, N);
    else
        GreekMat = lu.compute(Eigen::MatrixXd::Identity(2*N, 2*N) - J_rs).inverse()*J_a;
#endif
    //Eigen::MatrixXd::Map(&res[0], res_eigen.rows(), res_eigen.cols()) = res_eigen;
    //return res;
    jacobian_set = true;
}



Eigen::MatrixXd BlackScholesNetwork::get_delta_v1() const
{
    if(!jacobian_set) LOG(ERROR) << "Jacobian not computed before calling get_delta";
    const Eigen::MatrixXd res_eigen =  exprt * GreekMat * St.asDiagonal();
    return res_eigen;
}

//Eigen::MatrixXd BlackScholesNetwork::get_vega(const Eigen::VectorXd Z) const
Eigen::MatrixXd BlackScholesNetwork::get_vega(const Eigen::MatrixXd Z) const
{
    Eigen::MatrixXd res_eigen(2*N, N);
    if(!jacobian_set) LOG(ERROR) << "Jacobian not computed before calling get_vega";
    Eigen::MatrixXd sigma_m = sigma.array().sqrt().matrix().asDiagonal();
    res_eigen  = exprt* GreekMat * (((Z.array()*std::sqrt(T) - (T*sigma.array().sqrt()).array()).array() * St_full.array()).matrix());// * (S0.array()* St.array()).matrix().asDiagonal());
    return res_eigen;
}

Eigen::MatrixXd BlackScholesNetwork::get_theta(const Eigen::MatrixXd Z) const
{
    Eigen::MatrixXd res_eigen(2*N, 1);
    if(!jacobian_set) LOG(ERROR) << "Jacobian not computed before calling get_theta";
    const Eigen::ArrayXd tmp_deriv = r - sigma.array()/2.0 + sigma.array().sqrt()*Z.array()/(2.0*std::sqrt(T));
    //Eigen::MatrixXd res_eigen = GreekMat * (( exprt*tmp_deriv.array() * St.array()).matrix().asDiagonal()) - exprt*r*x;
    res_eigen = GreekMat * (( exprt*tmp_deriv * St.array()).matrix()) - exprt*r*x;
    return res_eigen;
}

Eigen::MatrixXd BlackScholesNetwork::get_rho() const
{
    if(!jacobian_set) LOG(ERROR) << "Jacobian not computed before calling get_rho";
    const Eigen::MatrixXd res_eigen = T*exprt* (GreekMat * (St_full).matrix() - x);
    return res_eigen;
}

Eigen::MatrixXd BlackScholesNetwork::get_pi() const
{
    if(!jacobian_set) LOG(ERROR) << "Jacobian not computed before calling get_rho";
    const Eigen::MatrixXd res_eigen = Eigen::MatrixXd(GreekMat).colwise().sum() - Eigen::RowVectorXd::Constant(N,1.0);
    return res_eigen;
}


Eigen::MatrixXd BlackScholesNetwork::get_scalar_allGreeks(const Eigen::Ref<const Eigen::MatrixXd>& Z) const
{
    if(!jacobian_set) LOG(ERROR) << "Jacobian not computed before calling Greeks";
    Eigen::MatrixXd res = Eigen::MatrixXd(4,2);
    Eigen::ArrayXd sigma_sqrt(sigma.size());
    sigma_sqrt = sigma.array().sqrt();
    const Eigen::MatrixXd res_delta =  (exprt * GreekMat * St.asDiagonal());
    const Eigen::MatrixXd res_vega = exprt* GreekMat * (((std::sqrt(T)*Z.array() - T*sigma_sqrt).array() * St_full.array()).matrix().asDiagonal());// * (S0.array()* St.array()).matrix().asDiagonal());
    const Eigen::ArrayXd tmp_deriv = (r - sigma.array()/2.0) +  sigma_sqrt*Z.array()/(2.0*std::sqrt(T));
    const Eigen::MatrixXd res_theta = GreekMat * (( exprt*tmp_deriv * St_full.array()).matrix().asDiagonal()) - exprt*r*x;
    const Eigen::MatrixXd res_rho = T*exprt* (GreekMat * (St_full).matrix() - x);
    res(0,0) = res_delta.topRows(N).sum();
    res(0,1) = res_delta.bottomRows(N).sum();
    res(1,0) = res_vega.topRows(N).sum();
    res(1,1) = res_vega.bottomRows(N).sum();
    res(2,0) = res_theta.topRows(N).sum();
    res(2,1) = res_theta.bottomRows(N).sum();
    res(3,0) = res_rho.topRows(N).sum();
    res(3,1) = res_rho.bottomRows(N).sum();

    return res;
}


void BlackScholesNetwork::debug_print()
{
    LOG(INFO) << "DEBUG PRINT BLACK SCHOLES NETWORK";
    Eigen::IOFormat CleanFmt(3, 0, " ", "\n", "[", "]");
#if USE_SPARSE_INTERNAL
    LOG(INFO) << "initialized: " << initialized << ", N = " << N <<", T = " << T << ", r = " << r << ", exprt = " << exprt << "M: \n" <<  Eigen::MatrixXd(M).format(CleanFmt);
#else
    LOG(INFO) << "initialized: " << initialized << ", N = " << N <<", T = " << T << ", r = " << r << ", exprt = " << exprt;
    LOG(INFO) << "M: \n" <<  M.format(CleanFmt);
    LOG(INFO) << "GreekMat: \n" <<  GreekMat.format(CleanFmt);
    LOG(INFO) << "Sigma: \n" <<  sigma.format(CleanFmt);
#endif
    LOG(INFO) << "\nS0 \n" << S0;
    LOG(INFO) << "\nSt: \n"  << St;
    LOG(INFO) << "\ndebt: \n" << debt;
    LOG(INFO) << "\nsigma: \n" << sigma;
}


