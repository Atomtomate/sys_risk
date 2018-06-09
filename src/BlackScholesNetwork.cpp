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


BlackScholesNetwork::BlackScholesNetwork(const Eigen::Ref<Mat>& M, const Eigen::Ref<Vec>& S0, const Eigen::Ref<Vec>& assets, const Eigen::Ref<Vec>& debt, const double T, const double r):
        M(M), N(M.rows()), S0(S0), St(assets), debt(debt), T(T), r(r), exprt(std::exp(-r * T))
{
    initialized = true;
    //EXPECT_EQ(M.cols(), 2*M.rows()) << "Dimensions for cross holding matrix invalid!";
    //EXPECT_EQ(assets.rows(), debt.rows()) <<  "Dimensions of debts and asset vector do not match!";
    //EXPECT_EQ(assets.rows(), M.rows()) << "Dimensions for assets vector and cross holding matrix to not match!";
};


void BlackScholesNetwork::set_solvent()
{
    solvent.resize(N);
    for(unsigned int i = 0; i < N; i++) {
        solvent(i) = 1*(x(i)+x(i+N) >= debt(i));
    }
}


const Eigen::MatrixXd BlackScholesNetwork::run_valuation(unsigned int iterations)
{
    if(!initialized) throw std::logic_error("attempting to solve uninitialized model!");
    int N = M.rows();
    x.resize(2*N);

    double dist = 99.;
    Eigen::MatrixXd a = S0.array()*St.array();
    for(unsigned int r = 0; r < iterations; r++) {
        Eigen::MatrixXd tmp = a + M*x;
        auto distV = x;
        x.head(N) = (tmp - debt).array().max(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        distV = distV - x;
        dist = distV.norm();
        if(dist < 1.0e-12)
            break;
    }
    set_solvent();
    return x;
}


const Eigen::MatrixXd BlackScholesNetwork::iJacobian_fx()
{
    Eigen::MatrixXd J(2*N, 2*N);
    //@TODO: replace this loop with stacked solvent matrix x matrix operation?
    //J.topRows(N) = M.array().colwise() * solvent.cast<double>().array();
    //J.bottomRows(N) = M.array().colwise() * (1-solvent.cast<double>().array());

    if(solvent.size() == M.rows()) {
        for(unsigned int i = 0; i < N; i++) {
            J.row(i) = solvent(i)*M.row(i);
            J.row(N+i) = (1-solvent(i))*M.row(i);
        }
        J = (Eigen::MatrixXd::Identity(2*N, 2*N) - J).inverse();
    } else {
        LOG(WARNING) << "solvent has wrong size: " << solvent.size();
        J = Eigen::MatrixXd::Zero(2*N,2*N);
    }
    return J;
}


const Eigen::MatrixXd BlackScholesNetwork::Jacobian_va()
{
    Eigen::MatrixXd J(2*N, N);
    if(solvent.size() == M.rows()) {
        Eigen::MatrixXd sd = solvent.asDiagonal();
        J << sd, (Eigen::MatrixXd::Identity(N, N) - sd);
    } else {
        LOG(WARNING) << "solvent has wrong size: " << solvent.size();
        J = Eigen::MatrixXd::Zero(2*N,N);
    }
    return J;
}


//@TODO: check if return Eigen::Refwould be better here
const Eigen::VectorXd BlackScholesNetwork::get_assets()
{
    return (S0.array()*St.array());
}
