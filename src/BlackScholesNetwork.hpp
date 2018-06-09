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

#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif

#include <boost/serialization/vector.hpp>
#include <boost/serialization/optional.hpp>


#include "ValuationConfig.h"
#include "StatAcc.hpp"


class BlackScholesNetwork
{
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;
private:
    double T;
    double r;
    unsigned int N;
    bool initialized;
    Mat M;
    Vec x;
    Vec S0;
    Vec St;
    Vec debt;
    Vec solvent;
    double exprt;

    void set_solvent();

    const Mat iJacobian_fx();

    const Mat Jacobian_va();

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
    BlackScholesNetwork(const Eigen::Ref<Mat>& M, const Eigen::Ref<Vec>& S0, const Eigen::Ref<Vec>& assets, const Eigen::Ref<Vec>& debt, const double T, const double r);



    /**
     * @brief               Finds the fixed point of the cross holding problem at maturity T.
     * @param iterations    maximum number of self consistency iterations.
     * @return              returns vector of value of debt and value of equity.
     */
    const Mat run_valuation(unsigned int iterations);


    inline void set_St(const Vec &st) {
        if(st.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and assets!");
        St = st;
    }

    inline void re_init(const Eigen::Ref<const Mat>& M_new, const Eigen::Ref<const Vec> &s0, const Eigen::Ref<const Vec> &d) {
        initialized = true;
        M = M_new;
        N = M.rows();
        if(s0.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and assets prefactor!");
        S0 = s0;
        St = s0;
        if(d.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and debts!");
        debt = d;
    }

    //@TODO: consistent return typex
    inline const Vec get_S0() const {
        return S0;
    }

    inline const Vec get_St() const {
        return St;
    }

    inline const Vec get_debt() const {
        return debt;
    }

    inline const Mat get_M() const {
        return M;
    }

    const Vec get_assets();

    //@TODO: move implementation to *.cpp
    const Vec get_rs() {
        return x;
       /* std::vector<double> ret;
        ret.resize(x.size());
        Eigen::VectorXd::Map(&ret[0], x.size()) = x;
        return ret;
        */
    }

    //inline const Eigen::VectorXd get_rs_eigen() const {
    //    return x;
    //}

    const Vec get_valuation() {
        Vec res = (x.head(N) + x.tail(N));
        return res;
        /*
        std::vector<double> ret;
        ret.resize(N);
        Eigen::VectorXd::Map(&ret[0], N) = x.head(N) + x.tail(N);
        return ret;*/
    }

   const Vec get_solvent() {
        return solvent;
        /*std::vector<double> res;
        res.resize(solvent.size());
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&res[0], solvent.size()) = solvent;
        return res;*/
    }

    const Mat get_delta_v1() {
        auto Jrs = iJacobian_fx();
        auto Jva = Jacobian_va();
        auto res_eigen =  exprt*(Jrs*Jva)*(St.asDiagonal());
        return res_eigen;
        //LOG(INFO) << "exprt:" << exprt <<"\n===\n" << St << "\n====\n";
        //LOG(ERROR) << Jrs << "\n\n" << Jva << "\n\n" << res_eigen << "\n\n";
        //std::vector<double> res;
        //res.resize(2*N*N);
        //EXPECT_EQ(res_eigen.rows(), 2*N) << "Number of rows for Delta computation incorrect";
        //Eigen::MatrixXd::Map(&res[0], res_eigen.rows(), res_eigen.cols()) = res_eigen;
        //return res;
    }

    //get_v_out()

};


#endif // SRC_MULTIVAR_BLACKSCHOLES_HPP_
