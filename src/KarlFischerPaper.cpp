#include "KarlFischerPaper.hpp"

void figure6(void)
{
    const unsigned int N = 2;
    const unsigned int nPoints = 50000;
    const double a = 1.0;                   // exogenously priced assets // mu = -0.5*sigma*sigma + ln(a)
    Eigen::MatrixXd Md(N,N);                // cross-owned debt factors     -- 0.1, 0.2 ... 0.9
    Eigen::MatrixXd Ms(N,N);                // cross-owned equity factors   -- 0.1, 0.2 ... 0.9
    Eigen::MatrixXd M(N, 2*N);
    Eigen::VectorXd V(N);                   // firm values                  -- 0.1, 0.2 ... 0.9
    Eigen::VectorXd debt(N);                // zero coupon debt
    Md << 0., 0.95, 0.95, 0.;
    Ms << 0., 0.95, 0.95, 0.;
    Md = Eigen::MatrixXd::Identity(N,N);
    Ms = Eigen::MatrixXd::Identity(N,N);
    M << Md, Ms;
    debt << 0.5*11.3, 0.5*11.3;                        // liabilities, 0.1 ... 3.0
    // initial lognormal distribution parameters
    // coeff of var for A_i  --  0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.31, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0 
    // 0.00995, 0.08618, 0.03922, 0.22314, 0.44629, 0.69315, 1., 1.17865, 1.60944, 1.98100, 2.30259, 3.25810, 4.04742, 4.61512
    Eigen::VectorXd sigma(N);
    sigma << 0.09, 0.09;//std::log(1.0*1.0 + 1.0), std::log(1.0*1.0 + 1.0);
    Eigen::VectorXd mu(N);
    mu << -0.5*sigma(0)*sigma(0) + std::log(a), -0.5*sigma(1)*sigma(1) + std::log(a);
    trng::yarn2 gen_v1;
    //LOG(INFO) << "lognormal distribution with mu = " << mu.transpose() << " and sigma = " << sigma.transpose();
    trng::lognormal_dist<> v1(mu(0), sigma(0));
    trng::lognormal_dist<> v2(mu(1), sigma(1)); 
    for(unsigned int i = 0; i < nPoints; i++)
    {
        V << v1(gen_v1), v2(gen_v1);
        //LOG(INFO) << V;
        auto res = run_modified(M, V, debt, N, 1000);
        std::cout << res(0) << "\t" << res(1) << std::endl;
    }
}
