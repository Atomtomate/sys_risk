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

        void set_solvent(void);
        void set_M_ER(const double p, const double val, char which_to_set);

    public:
        MultiVar_BlackScholes(Eigen::MatrixXd M, Eigen::VectorXd assets, Eigen::VectorXd debt, double T, double r);

        MultiVar_BlackScholes(double p, double val, char which_to_set, Eigen::VectorXd assets, Eigen::VectorXd debt);

        Eigen::VectorXd run_valuation(unsigned int iterations);

        void set_assets(Eigen::VectorXd a)
        {
            if(2*a.size() != M.rows())
                throw std::logic_error("Dimensions of new assets do not match network!");
            assets = a;
        }
        void set_debt(Eigen::VectorXd d)
        {
            if(2*d.size() != M.rows())
                throw std::logic_error("Dimensions of new assets do not match network!");
            debt = d;
        }
        Eigen::VectorXd get_valuation(void) { return x; };
        Eigen::VectorXi get_solvent(void){ return solvent; };
};



Eigen::VectorXd run_modified(const Eigen::MatrixXd& zij, const Eigen::VectorXd& exo_assets,const Eigen::VectorXd& debt, const unsigned int N, const unsigned int max_it);

enum class ACase {DD, DS, SD, SS, ERROR};

Eigen::VectorXi classify_solvent(Eigen::VectorXd& v, Eigen::VectorXd& debt);
ACase classify_paper(Eigen::MatrixXd& zij, Eigen::VectorXd& assets, Eigen::VectorXd& debt);
ACase classify(Eigen::VectorXd& v, Eigen::VectorXd& debt);

void run_greeks(void);
Eigen::MatrixXd jacobian_a(Eigen::VectorXd& solvent);
Eigen::MatrixXd jacobian_rs(Eigen::MatrixXd& M, Eigen::VectorXd& solvent);
#endif // SRC_MULTIVAR_BLACKSCHOLES_HPP_
