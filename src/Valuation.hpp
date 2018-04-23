#ifndef VALUATION_HPP_
#define VALUATION_HPP_

#include "easylogging++.h"
// Tina's Random Number Generator
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/lognormal_dist.hpp>
#include <trng/correlated_normal_dist.hpp>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>

#include "ValuationConfig.h"
#include "MCAcc.hpp"

#include <Eigen/Dense>


Eigen::MatrixXd run_valuation(Eigen::MatrixXd& vij, Eigen::MatrixXd& zij, Eigen::VectorXd& B, const unsigned int N, const unsigned int max_it, const unsigned int L);


Eigen::VectorXd run_modified(const Eigen::MatrixXd& zij, const Eigen::VectorXd& exo_assets,const Eigen::VectorXd& debt, const unsigned int N, const unsigned int max_it);

enum class ACase {DD, DS, SD, SS, ERROR};

Eigen::VectorXi classify_solvent(Eigen::VectorXd& v, Eigen::VectorXd& debt);
ACase classify_paper(Eigen::MatrixXd& zij, Eigen::VectorXd& assets, Eigen::VectorXd& debt);
ACase classify(Eigen::VectorXd& v, Eigen::VectorXd& debt);

void run_greeks(void);
Eigen::MatrixXd jacobian_a(Eigen::VectorXd& solvent);
Eigen::MatrixXd jacobian_rs(Eigen::MatrixXd& M, Eigen::VectorXd& solvent);
#endif
