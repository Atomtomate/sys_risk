/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */


#ifndef KFP_HPP_
#define KFP_HPP_

#include <iostream>
#include <sstream>

#include "BlackScholesNetwork.hpp"

enum class ACase {DD, DS, SD, SS, ERROR};


Eigen::VectorXi classify(Eigen::VectorXd& v, Eigen::VectorXd& debt);
ACase classify_paper(Eigen::VectorXd& v, Eigen::VectorXd& debt);

Eigen::VectorXd run_modified(const Eigen::MatrixXd& zij, const Eigen::VectorXd& exo_assets, const Eigen::VectorXd& debt, const unsigned int N, const unsigned int max_it);
void figure6();

#endif
