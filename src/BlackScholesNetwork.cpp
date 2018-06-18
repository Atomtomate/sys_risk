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
#endif

    //EXPECT_EQ(M.cols(), 2*M.rows()) << "Dimensions for cross holding matrix invalid!";
    //EXPECT_EQ(assets.rows(), debt.rows()) <<  "Dimensions of debts and asset vector do not match!";
    //EXPECT_EQ(assets.rows(), M.rows()) << "Dimensions for assets vector and cross holding matrix to not match!";
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
    { //for(unsigned int r = 0; r < iterations; r++) {
        auto tmp = (a + M*x);
        x_old = x;
        x.head(N) = (tmp - debt).array().max(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        dist = (x_old - x).norm();
        //if(dist < 1.0e-12)
        //    break;
    }
    set_solvent();
    return x;
}


//@TODO: check if return Eigen::Refwould be better here
const Eigen::VectorXd BlackScholesNetwork::get_assets()
{
    return (S0.array()*St.array());
}
