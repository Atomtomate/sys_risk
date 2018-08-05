/* Copyright (C) 7/30/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_GENRNDER_HPP
#define VALUATION_GENRNDER_HPP

#include "easylogging++.h"
// Tina's Random Number Generator
#include "trng/yarn2.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/lognormal_dist.hpp"
#include "trng/correlated_normal_dist.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <Eigen/SparseLU>


namespace Utils {

    void gen_basic_rejection(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p,
                             const double val, const int which_to_set);


    void gen_perm(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val,
                    const int which_to_set);

    void gen_sinkhorn(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val,
                       const int which_to_set);

    void gen_configuration_model(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val,
                       const int which_to_set);

    void gen_fixed_degree(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set);


    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd* M);

}

#endif //VALUATION_GENRNDER_HPP
