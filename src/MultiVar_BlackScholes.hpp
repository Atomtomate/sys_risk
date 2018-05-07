#ifndef SRC_MULTIVAR_BLACKSCHOLES_HPP_
#define SRC_MULTIVAR_BLACKSCHOLES_HPP_

#include <stdexcept>

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
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>

#include "ValuationConfig.h"
#include "MCAcc.hpp"

class MultiVar_BlackScholes
{
    private:
        double T;
        double r;
        Eigen::MatrixXd M;
        Eigen::VectorXd x;
        Eigen::VectorXd assets;
        Eigen::VectorXd debt;
        Eigen::VectorXi solvent;

        void set_solvent();
        void set_M_ER(double p, double val, char which_to_set);

    public:
        /**
         * @brief
         * @param M         Combined cross equity and cross debt matrix
         * @param assets    exogenous assets
         * @param debt      debts
         * @param T         maturity
         * @param r         interest rate
         */
        MultiVar_BlackScholes(Eigen::MatrixXd M, Eigen::VectorXd assets, Eigen::VectorXd debt, double T, double r);

        /**
         * @brief               Constructs the Black Scholes Model using random cross holdings.
         * @param p             Probability of cross holding
         * @param val           Value of cross holding (@TODO make this variable)
         * @param which_to_set  can be 0/1/2. 2: cross debt is 0, 1: cross equity is 0, 0: none is 0
         * @param assets        exogenous assets
         * @param debt          debts
         */
        MultiVar_BlackScholes(double p, double val, char which_to_set, Eigen::VectorXd assets, Eigen::VectorXd debt);

        /**
         * @brief               Finds the fixed point of the cross holding problem at maturity T.
         * @param iterations    maximum number of self consistency iterations.
         * @return              returns vector of value of debt and value of equity.
         */
        Eigen::VectorXd run_valuation(unsigned int iterations);

        void set_assets(const Eigen::VectorXd& a)
        {
            if(2*a.size() != M.rows())
                throw std::logic_error("Dimensions of new assets do not match network!");
            assets = a;
        }
        void set_debt(const Eigen::VectorXd& d)
        {
            if(2*d.size() != M.rows())
                throw std::logic_error("Dimensions of new assets do not match network!");
            debt = d;
        }
        Eigen::VectorXd get_rs() { return x; };
        Eigen::VectorXd get_valuation() { return x.head(M.rows()) + x.tail(M.rows()); };
        Eigen::VectorXi get_solvent(){ return solvent; };
};



Eigen::VectorXd run_modified(const Eigen::MatrixXd& zij, const Eigen::VectorXd& exo_assets,const Eigen::VectorXd& debt, unsigned int N, unsigned int max_it);

enum class ACase {DD, DS, SD, SS, ERROR};

Eigen::VectorXi classify_solvent(Eigen::VectorXd& v, Eigen::VectorXd& debt);
ACase classify_paper(Eigen::MatrixXd& zij, Eigen::VectorXd& assets, Eigen::VectorXd& debt);
ACase classify(Eigen::VectorXd& v, Eigen::VectorXd& debt);

void run_greeks(void);
Eigen::MatrixXd jacobian_a(Eigen::VectorXd& solvent);
Eigen::MatrixXd jacobian_rs(Eigen::MatrixXd& M, Eigen::VectorXd& solvent);
#endif // SRC_MULTIVAR_BLACKSCHOLES_HPP_
