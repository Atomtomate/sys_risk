/* Copyright (C) 9/25/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_STUDENTT_HPP
#define VALUATION_STUDENTT_HPP


#include <cmath>
#include "Eigen/Dense"
#include "easylogging++.h"

class Student_t_dist
{
private:
    double deg;
    int p;
    Eigen::MatrixXd sigma;
    Eigen::MatrixXd sigma_inv;
    Eigen::VectorXd mu;
    double log_prefactor;
    bool initialized;

public:
    Student_t_dist(int deg_):deg(deg_) { initialized = false;}

    Student_t_dist(Eigen::MatrixXd sigma_, Eigen::VectorXd mu_, int deg_):
            sigma(sigma_), mu(mu_), deg(deg_)
    {
        p = sigma_.rows();
        sigma_inv = sigma_.inverse();
        log_prefactor = std::lgamma((deg+p)*0.5) - std::lgamma(deg*0.5) - (p*0.5)*std::log(deg*M_PI) - 0.5*std::log(sigma.determinant());
        initialized = true;
    }

    Student_t_dist(Eigen::MatrixXd sigma_, int deg_):
            sigma(sigma_), deg(deg_)
    {
        p = sigma_.rows();
        sigma_inv = sigma_.inverse();
        log_prefactor = std::lgamma((deg+p)*0.5) - std::lgamma(deg*0.5) - (p*0.5)*std::log(deg*M_PI) - 0.5*std::log(sigma.determinant());
    }

    double logpdf(Eigen::VectorXd x)
    {
        Eigen::VectorXd xp = x - mu;
        double xm = xp.transpose()*sigma_inv*xp;
        return log_prefactor - 0.5*(deg+p)*std::log(1.0 + xm/deg);
    }

    double logpdf(Eigen::VectorXd mu, Eigen::VectorXd x)
    {
        Eigen::VectorXd xp = x - mu;
        double xm = xp.transpose()*sigma_inv*xp;
        return log_prefactor - 0.5*(deg+p)*std::log(1.0 + xm/deg);
    }

    double logpdf(Eigen::MatrixXd sigma, Eigen::VectorXd mu, Eigen::VectorXd x)
    {
        p = sigma.rows();
        Eigen::MatrixXd sigma_inv = sigma.inverse();
        double log_pre = std::lgamma((deg+p)*0.5) - std::lgamma(deg*0.5) - (p*0.5)*std::log(deg*M_PI) - 0.5*std::log(sigma.determinant());
        Eigen::VectorXd xp = x - mu;
        double xm = xp.transpose()*sigma_inv*xp;
        return log_pre - 0.5*(deg+p)*std::log(1.0 + xm/deg);
    }
};



#endif //VALUATION_STUDENTT_HPP
