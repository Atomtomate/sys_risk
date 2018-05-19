#ifndef SRC_BLACKSCHOLES_NETWORK_HPP_
#define SRC_BLACKSCHOLES_NETWORK_HPP_

#include <stdexcept>
#include <algorithm>

#include "gtest/gtest.h"
#include "easylogging++.h"
// Tina's Random Number Generator
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/lognormal_dist.hpp>
#include <trng/correlated_normal_dist.hpp>
#include <Eigen/Dense>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/optional.hpp>

#include <gtest/gtest.h>

#include "ValuationConfig.h"
#include "StatAcc.hpp"


class BlackScholesNetwork
{
    public:
        double T;
        double r;
        static size_t gbl_dbg_counter;
        size_t dbg_counter;
        Eigen::MatrixXd M;
        Eigen::VectorXd x;
        Eigen::VectorXd S0;
        Eigen::VectorXd St;
        Eigen::VectorXd debt;
        Eigen::VectorXd solvent;
        double exprt;

        void set_solvent();
        void set_M_ER(double p, double val, char which_to_set);
        Eigen::MatrixXd iJacobian_fx();
        Eigen::MatrixXd Jacobian_va();

    public:
        BlackScholesNetwork(const BlackScholesNetwork&) = delete;
        BlackScholesNetwork& operator=(const BlackScholesNetwork&) = delete;

        /**
         * @brief
         * @param T         maturity
         * @param r         interest rate
         */
        BlackScholesNetwork(const Eigen::MatrixXd& M,const double T,const double r);

        /**
         * @brief
         * @param M         Combined cross equity and cross debt matrix
         * @param assets    exogenous assets
         * @param debt      debts
         * @param T         maturity
         * @param r         interest rate
         */
        BlackScholesNetwork(Eigen::MatrixXd& M, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r);

        /**
         * @brief               Constructs the Black Scholes Model using random cross holdings.
         * @param p             Probability of cross holding
         * @param val           Value of cross holding (@TODO make this variable)
         * @param which_to_set  can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
         * @param assets        exogenous assets
         * @param debt          debts
         * @param T         maturity
         * @param r         interest rate
         */
        BlackScholesNetwork(double p, double val, char which_to_set, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r);

        /**
         * @brief               Finds the fixed point of the cross holding problem at maturity T.
         * @param iterations    maximum number of self consistency iterations.
         * @return              returns vector of value of debt and value of equity.
         */
        std::vector<double> run_valuation(unsigned int iterations);

        inline void get_S0(const Eigen::VectorXd& s0)
        {
            EXPECT_EQ(S0.size(), M.rows()) << "Dimensions of new assets do not match network dimensions!";
            S0 = s0;
        }

        inline void set_S0(const Eigen::VectorXd& s0)
        {
            EXPECT_EQ(s0.size(), M.rows()) << "Dimensions of new assets do not match network dimensions!";
            S0 = s0;
        }

        inline void set_St(const Eigen::VectorXd& a)
        {
            EXPECT_EQ(a.size(), M.rows()) << "Dimensions of new assets do not match network dimensions!";
            St = a;
        }

        inline void set_debt(const Eigen::VectorXd& d)
        {
            EXPECT_EQ(d.size(), M.rows()) << "Dimensions of new debts do not match network dimensions!";
            debt = d;
        }

        inline void set_M(Eigen::MatrixXd M_new) {M = M_new;}

        //@TODO: move implementation to *.cpp
        std::vector<double> get_assets();

        auto get_rs()
        {
            std::vector<double> ret;
            ret.resize(x.size());
            Eigen::VectorXd::Map(&ret[0], x.size()) = x;
            return ret;
        }

        auto get_valuation()
        {
            const auto n = M.rows();
            std::vector<double> ret;
            ret.resize(n);
            Eigen::VectorXd::Map(&ret[0], n) = x.head(n) + x.tail(n);
            return ret;
        }
        auto get_solvent()
        {
            std::vector<double> res;
            res.resize(solvent.size());
            Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&res[0], solvent.size()) = solvent;
            return res;
        }
    //delta_pw = delta_pw + std::exp(-r*T)*(Jrs_m1*Ja)*(St.asDiagonal());
    //delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
        std::vector<double> get_delta_v1()
        {
            const auto N = M.rows();
            auto Jrs = iJacobian_fx();
            auto Jva = Jacobian_va();
            LOG(ERROR) << Jrs;
            LOG(ERROR) << Jva;
            auto res_eigen =  exprt*(Jrs*Jva)*(St.asDiagonal());
            LOG(INFO) << res_eigen;
            std::vector<double> res;
            res.resize(2*N*N);
            EXPECT_EQ(res_eigen.rows(), 2*N) << "Number of rows for Delta computation incorrect";
            EXPECT_EQ(res_eigen.cols(), N) << "Number of cols for Delta computation incorrect";
            Eigen::MatrixXd::Map(&res[0], res_eigen.rows(), res_eigen.cols()) = res_eigen;
            return res;
        }

        auto get_delta_v2()
        {
            const auto N = M.rows();
            //Eigen::VectorXd ln_fac = (itSigma*Z).array()/S0.array();
            auto res = Eigen::MatrixXd::Zero(2*N,N);
            return res;
        }

};





#endif // SRC_MULTIVAR_BLACKSCHOLES_HPP_
