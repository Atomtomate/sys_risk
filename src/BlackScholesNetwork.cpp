/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#include "BlackScholesNetwork.hpp"

size_t BlackScholesNetwork::gbl_dbg_counter = 0;

BlackScholesNetwork::BlackScholesNetwork(const double T, const double r):
        T(T), r(r), exprt(std::exp(-r * T)), dbg_counter(gbl_dbg_counter)
{
    gbl_dbg_counter += 1;
    initialized = false;
    //EXPECT_EQ(M.cols(), 2 * M.rows()) << "Dimensions for cross holding matrix invalid!";
}


BlackScholesNetwork::BlackScholesNetwork(const Eigen::MatrixXd& M, const Eigen::VectorXd& S0, const Eigen::VectorXd& assets, const Eigen::VectorXd& debt, const double T, const double r):
        M(M), N(M.rows()), S0(S0), St(assets), debt(debt), T(T), r(r), exprt(std::exp(-r * T)),
        dbg_counter(gbl_dbg_counter)
{
    gbl_dbg_counter += 1;
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


std::vector<double> BlackScholesNetwork::run_valuation(unsigned int iterations)
{
    if(!initialized) throw std::logic_error("attempting to solve uninitialized model!");
    x = Eigen::VectorXd::Zero(2*N);
    Eigen::VectorXd a = S0.array()*St.array();
    double dist = 99.;
    for(unsigned int r = 0; r < iterations; r++) {
        auto tmp = a + M*x;
        auto distV = x;
        x.head(N) = (tmp - debt).cwiseMax(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        distV = distV - x;
        dist = distV.norm();
        if(dist < 1.0e-14)
            break;
    }
    set_solvent();
    return get_rs();
}


Eigen::MatrixXd BlackScholesNetwork::iJacobian_fx()
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
        LOG(WARNING) << "solvent has wrong size: " << solvent.size() << "\nobject id = " << dbg_counter;
        J = Eigen::MatrixXd::Zero(2*N,2*N);
    }
    return J;
}


Eigen::MatrixXd BlackScholesNetwork::Jacobian_va()
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


std::vector<double>  BlackScholesNetwork::get_assets()
{
    std::vector<double> res;
    res.resize(St.size());
    Eigen::VectorXd::Map(&res[0], St.size()) = S0.array()*St.array();
    return res;
}
