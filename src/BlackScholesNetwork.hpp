/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef SRC_BLACKSCHOLES_NETWORK_HPP_
#define SRC_BLACKSCHOLES_NETWORK_HPP_

#include <stdexcept>
#include <algorithm>

#include "easylogging++.h"
// Tina's Random Number Generator
#include "trng/yarn2.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/lognormal_dist.hpp"
#include "trng/correlated_normal_dist.hpp"
#include "Eigen/Dense"

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/optional.hpp>


#include "ValuationConfig.h"
#include "StatAcc.hpp"


class BlackScholesNetwork
{
private:
    double T;
    double r;
    unsigned int N;
    static size_t gbl_dbg_counter;
    size_t dbg_counter;
    bool initialized;
    Eigen::MatrixXd M;
    Eigen::VectorXd x;
    Eigen::VectorXd S0;
    Eigen::VectorXd St;
    Eigen::VectorXd debt;
    Eigen::VectorXd solvent;
    double exprt;

    void set_solvent();

    Eigen::MatrixXd iJacobian_fx();

    Eigen::MatrixXd Jacobian_va();

public:
    BlackScholesNetwork(const BlackScholesNetwork&) = delete;

    /* BlackScholesNetwork& operator=(const BlackScholesNetwork& rhs) = delete;
    {
        T = rhs.T;
        r = rhs.r;
        N = rhs.N;
        M = rhs.M;
        x = rhs.x;
        S0 = rhs.S0;
        St = rhs.St;
        debt = rhs.debt;
        solvent = rhs.solvent;
        exprt = rhs.exprt;
    }*/

    //BlackScholesNetwork()
    //{
        //LOG(WARNING) << "Default constructor for BlackScholesNetwork used. This could be unintentional.";
    //    initialized = false;
    //}

    /**
     * @brief
     * @param T         maturity
     * @param r         interest rate
     */
    BlackScholesNetwork(const double T,const double r);

    /**
     * @brief
     * @param M         Combined cross equity and cross debt matrix
     * @param assets    exogenous assets
     * @param debt      debts
     * @param T         maturity
     * @param r         interest rate
     */
    BlackScholesNetwork(const Eigen::MatrixXd& M, const Eigen::VectorXd& S0, const Eigen::VectorXd& assets, const Eigen::VectorXd& debt, const double T, const double r);


    //BlackScholesNetwork(double p, double val, char which_to_set, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r);

    /**
     * @brief               Finds the fixed point of the cross holding problem at maturity T.
     * @param iterations    maximum number of self consistency iterations.
     * @return              returns vector of value of debt and value of equity.
     */
    std::vector<double> run_valuation(unsigned int iterations);


    inline void set_St(const Eigen::VectorXd &st) {
        if(st.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and assets!");
        St = st;
    }

    inline void re_init(const Eigen::MatrixXd& M_new, const Eigen::VectorXd &s0, const Eigen::VectorXd &d) {
        initialized = true;
        M = M_new;
        N = M.rows();
        if(s0.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and assets prefactor!");
        S0 = s0;
        if(d.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and debts!");
        debt = d;
    }

    //@TODO: consistent return typex
    inline const Eigen::VectorXd& get_S0() const {
        return S0;
    }

    inline const Eigen::VectorXd& get_St() const {
        return St;
    }

    inline const Eigen::VectorXd& get_debt() const {
        return debt;
    }

    inline const Eigen::MatrixXd& get_M() const {
        return M;
    }

    std::vector<double> get_assets();

    //@TODO: move implementation to *.cpp
    auto get_rs() {
        std::vector<double> ret;
        ret.resize(x.size());
        Eigen::VectorXd::Map(&ret[0], x.size()) = x;
        return ret;
    }

    inline const Eigen::VectorXd get_rs_eigen() const {
        return x;
    }

    auto get_valuation() {
        std::vector<double> ret;
        ret.resize(N);
        Eigen::VectorXd::Map(&ret[0], N) = x.head(N) + x.tail(N);
        return ret;
    }

    auto get_solvent() {
        std::vector<double> res;
        res.resize(solvent.size());
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&res[0], solvent.size()) = solvent;
        return res;
    }

    std::vector<double> get_delta_v1() {
        auto Jrs = iJacobian_fx();
        auto Jva = Jacobian_va();
        auto res_eigen =  exprt*(Jrs*Jva)*(St.asDiagonal());
        //LOG(INFO) << "exprt:" << exprt <<"\n===\n" << St << "\n====\n";
        //LOG(ERROR) << Jrs << "\n\n" << Jva << "\n\n" << res_eigen << "\n\n";
        std::vector<double> res;
        res.resize(2*N*N);
        //EXPECT_EQ(res_eigen.rows(), 2*N) << "Number of rows for Delta computation incorrect";
        Eigen::MatrixXd::Map(&res[0], res_eigen.rows(), res_eigen.cols()) = res_eigen;
        return res;
    }


};


#endif // SRC_MULTIVAR_BLACKSCHOLES_HPP_
