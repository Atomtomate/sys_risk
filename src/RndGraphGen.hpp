/* Copyright (C) 7/30/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_GENRNDER_HPP
#define VALUATION_GENRNDER_HPP

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


namespace Utils {

    /*!
     * @brief               Generates matrix from ER model and rejects if Sinkhorn does not converge
     * @param M             Adjacency matrix
     * @param gen_u         0-1-uniform generator
     * @param p             probability of edge (ER model)
     * @param val           sum_i M_ij
     * @param which_to_set  1 equity (left), 2 debt (right), 0 both
     */
    void gen_basic_rejection(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p,
                             const double val, const int which_to_set);

    /*!
     * @brief               Generates matrix from ER model and does some initial checks before applying sinkhorn
     * @param M             Adjacency matrix
     * @param gen_u         0-1-uniform generator
     * @param p             probability of edge (ER model)
     * @param val           sum_i M_ij
     * @param which_to_set  1 equity (left), 2 debt (right), 0 both
     */
    void gen_sinkhorn(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val,
                       const int which_to_set);

    /*!
     * @brief               Generates matrix according to configuration model with fixed degree sequence (all N*p)
     * @param M             Adjacency matrix
     * @param gen_u         0-1-uniform generator
     * @param p             "probability" of edge. Fixed degree N*p
     * @param val           sum_i M_ij
     * @param which_to_set  1 equity (left), 2 debt (right), 0 both
     */
    void gen_configuration_model(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val,
                       const int which_to_set);

    /*!
     * @brief               Generates matrix with fixed degree according to algorithm in thesis
     * @param M             Adjacency matrix
     * @param gen_u         0-1-uniform generator
     * @param p             "probability" of edge. Fixed degree N*p
     * @param val           sum_i M_ij
     * @param which_to_set  1 equity (left), 2 debt (right), 0 both
     */
    void gen_fixed_degree(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set);


    // helper functions
    /*!
     * @brief
     * @param M
     * @return
     */
    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd* M);
    Eigen::MatrixXd avg_row_col_sums(Eigen::MatrixXd* M);

    int fixed_degree(Eigen::MatrixXd* M);

    std::pair<double,double> avg_io_deg(Eigen::MatrixXd* M);


    void fixed_2d(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set);

    void gen_ring(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set);

    void gen_star(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set);

    void gen_uniform(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set);
}

#endif //VALUATION_GENRNDER_HPP
