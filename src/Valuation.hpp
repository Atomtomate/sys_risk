#ifndef VALUATION_HPP_
#define VALUATION_HPP_

#include "easylogging++.h"
// Tina's Random Number Generator
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "ValuationConfig.h"

#include <Eigen/Dense>


Eigen::MatrixXd run_valuation(Eigen::MatrixXd& vij, Eigen::MatrixXd& zij, Eigen::VectorXd& B, const unsigned int N, const unsigned int max_it, const unsigned int L);

#endif
