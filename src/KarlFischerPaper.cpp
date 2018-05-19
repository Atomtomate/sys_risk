#include "KarlFischerPaper.hpp"

Eigen::VectorXd run_modified(const Eigen::MatrixXd& zij, const Eigen::VectorXd& exo_assets,const Eigen::VectorXd& debt, const unsigned int N, const unsigned int max_it)
{
    Eigen::VectorXd Zl = Eigen::VectorXd::Zero(2*N);
    double dist = 99.;
    for(unsigned int r = 0; r < max_it; r++)
    {
        auto tmp = exo_assets + zij*Zl;
        auto distV = Zl;
        Zl.head(N) = (tmp - debt).cwiseMax(0.);
        Zl.tail(N) = tmp.cwiseMin(debt);
        distV = distV - Zl;
        dist = distV.norm();
        if(dist < 1.0e-14)
            return Zl;
    }
    return Zl;
}

Eigen::VectorXi classify(Eigen::VectorXd& v, Eigen::VectorXd& debt)
{
Eigen::VectorXi res(v.size());
for(unsigned int i = 0; i < v.size(); i++)
{
res(i) = 1*(v(i) >= debt(i));
}
return res;
}



ACase classify_paper(Eigen::VectorXd& v, Eigen::VectorXd& debt)
{
ACase res = ACase::ERROR;
if((v(0) >= debt(0)) && (v(1) >= debt(1)))
{
res = ACase::SS;
}
else if((v(0) >= debt(0)) && (v(1) < debt(1)))
{
res = ACase::SD;
}
else if((v(0) < debt(0)) && (v(1) >= debt(1)))
{
res = ACase::DS;
}
else if((v(0) < debt(0)) && (v(1) < debt(1)))
{
res = ACase::DD;
}
else
{
LOG(ERROR) << "assets outside Suzuki regions, this is a bug.";
}
return res;
}

void figure6()
{
    const unsigned int N = 2;
    const unsigned int nPoints = 10000;
    const double a = 1.0;                   // exogenously priced assets // mu = -0.5*sigma*sigma + ln(a)
    Eigen::MatrixXd Md(N,N);                // cross-owned debt factors     -- 0.1, 0.2 ... 0.9
    Eigen::MatrixXd Ms(N,N);                // cross-owned equity factors   -- 0.1, 0.2 ... 0.9
    Eigen::MatrixXd M(N, 2*N);
    Eigen::VectorXd V(N);                   // firm values                  -- 0.1, 0.2 ... 0.9
    Eigen::VectorXd debt(N);                // zero coupon debt
    Md << 0.00, 0.95, 0.95, 0.00;
    Ms << 0.00, 0.95, 0.95, 0.00;
    M << Ms, Md;
    debt << 11.3, 11.3;//0.05*11.3, 0.01*11.3;                        // liabilities, 0.1 ... 3.0
    // initial lognormal distribution parameters
    // coefficient of var for A_i  --  0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.31, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0
    // 0.00995, 0.08618, 0.03922, 0.22314, 0.44629, 0.69315, 1., 1.17865, 1.60944, 1.98100, 2.30259, 3.25810, 4.04742, 4.61512
    Eigen::VectorXd sigma(N);
    sigma << 1, 1;//std::log(1.0*1.0 + 1.0), std::log(1.0*1.0 + 1.0);
    Eigen::VectorXd mu(N);
    mu << -0.50*sigma(0)*sigma(0) + std::log(a), -0.50*sigma(1)*sigma(1) + std::log(a);
    trng::yarn2 gen_v1;
    //LOG(INFO) << "lognormal distribution with mu = " << mu.transpose() << " and sigma = " << sigma.transpose();
    trng::lognormal_dist<> v1(mu(0), sigma(0));
    trng::lognormal_dist<> v2(mu(1), sigma(1)); 
    std::stringstream ss;
    for(unsigned int i = 0; i < nPoints; i++)
    {
        V << v1(gen_v1), v2(gen_v1);
        //LOG(INFO) << V;
        auto rs = run_modified(M, V, debt, N, 1000);
        Eigen::VectorXd v_res(N);
        for(unsigned int j = 0; j < N; j++ )
        {
            v_res(j) = rs(j) + rs(j+N);
        }
        ss << v_res(0) << "\t" << v_res(1) << "\t" << static_cast<std::underlying_type<ACase>::type>(classify_paper(v_res, debt)) << std::endl;
    }
    std::cout << ss.str();
}
