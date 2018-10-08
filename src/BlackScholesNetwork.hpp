/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef SRC_BLACKSCHOLES_NETWORK_HPP_
#define SRC_BLACKSCHOLES_NETWORK_HPP_

#define USE_SPARSE_INTERNAL 1


#include <stdexcept>
#include <algorithm>

#include "easylogging++.h"
// Tina's Random Number Generator
#include "trng/yarn2.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/lognormal_dist.hpp"
#include "trng/correlated_normal_dist.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <Eigen/SparseLU>

#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif

#include <boost/serialization/vector.hpp>
#include <boost/serialization/optional.hpp>


#include "ValuationConfig.h"
#include "StatAcc.hpp"
#include "RndGraphGen.hpp"

struct simulationParameters
{
    int N;
    double T;
    double r;
    double sigma;
    double S0;
    double conn;
    double colSums;
    double defaultScale;
    int which_to_set;
};


class BlackScholesNetwork
{
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;
private:
    double T;
    double r;
    int N;
    bool initialized;
    Vec x;
    Vec S0;
    Vec St;
    Vec St_full;
    Vec debt;
    Vec solvent;
    Vec sigma;
    double exprt;
    //Mat sigma_diag;
    bool jacobian_set;
#if USE_SPARSE_INTERNAL
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> lu;
    Eigen::SparseMatrix<double, Eigen::ColMajor> Id;
    Eigen::SparseMatrix<double, Eigen::ColMajor> M;
    //Eigen::SparseMatrix<double, Eigen::ColMajor> GreekMat;
    //Eigen::SparseMatrix<double, Eigen::ColMajor> Jrs;
    //Eigen::SparseMatrix<double, Eigen::ColMajor> J_a;
    //Eigen::SparseMatrix<double, Eigen::ColMajor> Z;
#else
    Mat M;
    //Mat J_a;
    Mat Id;
    Eigen::PartialPivLU<Eigen::MatrixXd> lu;
#endif
    Mat GreekMat;

    void set_solvent();



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
    BlackScholesNetwork(const Eigen::Ref<Vec>& S0, const Eigen::Ref<Vec>& debt, const Eigen::Ref<Vec>& sigma_, const double T,const double r);

    /**
     * @brief
     * @param M         Combined cross equity and cross debt matrix
     * @param assets    exogenous assets
     * @param debt      debts
     * @param T         maturity
     * @param r         interest rate
     */
    BlackScholesNetwork(const Eigen::Ref<Mat>& M, const Eigen::Ref<Vec>& S0, const Eigen::Ref<Vec>& assets, const Eigen::Ref<Vec>& debt, const Eigen::Ref<Vec>& sigma_, const double T, const double r);



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
        St_full = S0.array()*St.array();
    }

    void re_init(const Eigen::Ref<const Mat>& M_new)
    {
        if(M_new.rows() != S0.size()) throw std::logic_error("re-initialized with wrong M size!");
#if USE_SPARSE_INTERNAL
        M = M_new.sparseView();
        M.makeCompressed();
#else
        M = M_new;
#endif
    }

    void re_init(const Eigen::Ref<const Mat>& M_new, const Eigen::Ref<const Vec> &s0, const Eigen::Ref<const Vec> &d, const Eigen::Ref<const Vec> &sigma_) {
        N = M_new.rows();
        //Jrs.resize(2*N, 2*N);
        //J_a.resize(2*N, N);
        x.resize(2*N);
        x = Eigen::VectorXd::Zero(2*N);
#if USE_SPARSE_INTERNAL
        M = M_new.sparseView();
        M.makeCompressed();
        Id.resize(2*N, 2*N);
        Id.setIdentity();
#else
        M = M_new;
        lu = Eigen::PartialPivLU<Eigen::MatrixXd>(2*N);
        Id.resize(2*N, 2*N);
        Id.setIdentity();
#endif
        St.resize(N);
        St_full.resize(N);
        St = Eigen::VectorXd::Constant(N, 1.0);
        if(s0.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and assets prefactor!");
        S0 = s0;
        sigma.resize(N);
        sigma = sigma_;
        if(d.size() != N)
            throw std::logic_error("Mismatch between cross ownership matrix and debts!");
        debt = d;
        jacobian_set = false;
        initialized = true;
    }

    //@TODO: consistent return typex
    inline const Vec get_S0() const {
        return S0;
    }

    inline const Vec get_St() const {
        LOG(WARNING) << "St does NOT contain S0!";
        return St;
    }

    inline const Vec get_debt() const {
        return debt;
    }

    inline const Mat get_M() const {
#if USE_SPARSE_INTERNAL
        return Eigen::MatrixXd(M);
#else
        return M;
#endif
    }

    const Vec get_assets();

    //@TODO: move implementation to *.cpp
    const Vec get_rs() {
        return x;
    }

    const Vec get_valuation() {
        Vec res = (x.head(N) + x.tail(N));
        return res;
    }

   const Vec get_solvent() {
        return solvent;
   }


    void set_jacobian();

    Mat get_delta_v1() const;

    Mat get_vega(const Eigen::MatrixXd Z) const;

    Mat get_theta(const Eigen::MatrixXd Z) const;

    Mat get_rho() const;
    /*
    std::vector<double> ret;
    ret.resize(N);
    Eigen::VectorXd::Map(&ret[0], N) = x.head(N) + x.tail(N);
    return ret;*/


    Eigen::MatrixXd get_pi() const;

    Eigen::MatrixXd get_scalar_allGreeks(const Eigen::Ref<const Mat>& Z) const;

    void debug_print();


};
#endif // SRC_MULTIVAR_BLACKSCHOLES_HPP_
