/* Copyright (C) 9/25/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_MVARNORMAL_HPP
#define VALUATION_MVARNORMAL_HPP

#include "easylogging++.h"

class Multivariate_Normal_Dist
{
private:
    Eigen::MatrixXd sigma;
    Eigen::MatrixXd sigma_inv;
    Eigen::VectorXd mu;
    double log_prefactor;
    bool initialized;

public:
    Multivariate_Normal_Dist() { initialized = false;}

    Multivariate_Normal_Dist(Eigen::MatrixXd sigma_, Eigen::VectorXd mu_):
            sigma(sigma_), mu(mu_)
    {
        sigma_inv = sigma_.inverse();
        log_prefactor = -0.5*sigma.rows()*std::log(2.0*M_PI) - 0.5*std::log(sigma.determinant());
        initialized = true;
    }

    Multivariate_Normal_Dist(Eigen::MatrixXd sigma_):
            sigma(sigma_)
    {
        sigma_inv = sigma_.inverse();
        log_prefactor = -0.5*sigma.rows()*std::log(2.0*M_PI) - 0.5*std::log(sigma.determinant());
        initialized = true;
    }


    double logpdf(Eigen::VectorXd x)
    {
        Eigen::VectorXd xp = x - mu;
        double xm = xp.transpose()*sigma_inv*xp;
        return log_prefactor - 0.5*xm;
    }

    double logpdf(Eigen::VectorXd mu, Eigen::VectorXd x)
    {
        Eigen::VectorXd xp = x - mu;
        double xm = xp.transpose()*sigma_inv*xp;
        return log_prefactor - 0.5*xm;
    }

    double logpdf(Eigen::MatrixXd sigma, Eigen::VectorXd mu, Eigen::VectorXd x)
    {
        Eigen::MatrixXd sigma_inv = sigma.inverse();
        double log_pre = -0.5*sigma.rows()*std::log(2.0*M_PI) - 0.5*std::log(sigma.determinant());
        Eigen::VectorXd xp = x - mu;
        double xm = xp.transpose()*sigma_inv*xp;
        return log_pre - 0.5*xm;
    }
};




#endif //VALUATION_MVARNORMAL_HPP
