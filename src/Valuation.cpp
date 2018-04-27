#include "Valuation.hpp"

Eigen::MatrixXd run_valuation(Eigen::MatrixXd& vij, Eigen::MatrixXd& zij, Eigen::VectorXd& B, const unsigned int N, const unsigned int max_it, const unsigned int L)
{
    Eigen::VectorXd Z(2*N);
    Z = Eigen::VectorXd::Zero(2*N);
    for(unsigned int li = 0; li < L; li++)
    {
        Eigen::VectorXd Zl(2*N);
        Zl = Eigen::VectorXd::Random(2*N);
        Eigen::VectorXd V(N);
        V = Eigen::VectorXd::Random(N);
        double dist = 99;
        for(unsigned int r = 0; r < max_it; r++)
        {
            auto tmp = vij*V + zij*Zl;
            auto distV = Zl;
            Zl.head(N) = (tmp-B).cwiseMax(0.);
            Zl.tail(N) = tmp.cwiseMin(B);
            distV = distV - Zl; 
            dist = distV.norm();
            if(dist < 1.0e-06)
            {
                VLOG(4) << "converged with distance " << dist << " after " << r << " iterations.";
                break;
            }
        }
        Z += Zl;
    }
    Z = Z/L;
    return Z;
}

ACase classify_paper(Eigen::MatrixXd& zij, Eigen::VectorXd& assets, Eigen::VectorXd& debt)
{
    ACase res = ACase::ERROR;
    // ss
    if(( assets(0) + zij(0,3)*assets(1) >= (1. - zij(0,3)*zij(1,0))*debt(0) + (zij(0,3) - zij(0,1))*debt(1)    ) && (
    zij(1,2)*assets(0) + assets(1) >= (zij(1,2) - zij(1,0))*debt(0) + (1. - zij(1,2)*zij(0,1))*debt(1)))
    {
        res = ACase::SS;
    }
    else if((       //sd
        assets(0) + zij(0,1)*assets(1) >= (1. - zij(0,1)*zij(1,0))*debt(0)
        ) && (
        zij(1,2)*assets(0) + assets(1) < (zij(1,2) - zij(1,0))*debt(0) + (1. - zij(1,2)*zij(0,1))*debt(1)))
    {
        res = ACase::SD;
    }
    else if((       // ds
        assets(0) + zij(0,3)*assets(1) < (1. - zij(0,3)*zij(1,0))*debt(0) + (zij(0,3) - zij(0,1))*debt(1)
        ) && (
        zij(1,0)*assets(0) + assets(1) >= (1. - zij(1,0)*zij(0,1))*debt(1)))
    {
        res = ACase::DS;
    }
    else if((       // dd
        assets(0) + zij(0,1)*assets(1) < (1. - zij(0,1)*zij(1,0))*debt(0)
        ) && (
        zij(1,0)*assets(0) + assets(1) < (1. - zij(1,0)*zij(0,1))*debt(1)))
    {
        res = ACase::DD;
    }
    else
    {
        LOG(ERROR) << "assets outside Suzuki regions, this is a bug.";
    }
    return res;
}


Eigen::VectorXi classify_solvent(Eigen::VectorXd& v, Eigen::VectorXd& debt)
{
    Eigen::VectorXi res(v.size());
    for(unsigned int i = 0; i < v.size(); i++)
    {
        res(i) = 1*(v(i) >= debt(i));
    }
    return res;
}


ACase classify(Eigen::VectorXd& v, Eigen::VectorXd& debt)
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



Eigen::MatrixXd jacobian_a(Eigen::VectorXi& solvent)
{
    const unsigned int n = solvent.size();
    auto oneVec = Eigen::RowVectorXd::Ones(n);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2*n,n); 
    for(unsigned int i = 0; i < n; i++)
    {
        if(solvent(i) > 0)
        {
            J.row(i) = oneVec;
        }
        else
        {
            J.row(n+i) = oneVec;
        }
    }
    return J;
}

/**
 *  Md , Ms
 */
Eigen::MatrixXd jacobian_rs(Eigen::MatrixXd& M, Eigen::VectorXi& solvent)
{
    const unsigned int n = solvent.size();
    auto zeroVec = Eigen::RowVectorXd::Zero(2*n);
    Eigen::MatrixXd J(2*n,2*n);
    J << M, M;
    for(unsigned int i = 0; i < n; i++)
    {
        if(solvent(i) > 0)
        {
            J.row(n+i) = zeroVec;
        }
        else
        {
            J.row(i) = zeroVec;
        }
    }
    return J;
}

Eigen::VectorXd run_modified(const Eigen::MatrixXd& zij, const Eigen::VectorXd& exo_assets,const Eigen::VectorXd& debt, const unsigned int N, const unsigned int max_it)
{
    Eigen::VectorXd Zl(2*N);
    Zl.setZero(2*N);
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


void run_greeks(void)
{
    trng::yarn2 gen_v1;             // prng
    const unsigned N = 2;           // number of firms
    const unsigned nPoints = 5000;  // number of data points
    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(N,N);
    Eigen::MatrixXd eye2(N,2*N);
    Eigen::MatrixXd eye22 = Eigen::MatrixXd::Identity(2*N, 2*N);
    eye2 << eye, eye;
    Eigen::MatrixXd Md(N,N);                                // cross debt
    Eigen::MatrixXd Ms(N,N);                                // cross holdings
    Eigen::MatrixXd M(N,2*N);                               // Md, Ms
    Eigen::VectorXd Vb(N);                                  // lognormal distr equity, without a_0
    Eigen::VectorXd V(N);                                   // equity
    Eigen::VectorXd V_out(N);                               // outside investors
    Eigen::VectorXd debt(N);                                // debt
    Eigen::MatrixXd sigma(N,N);                             
    Eigen::MatrixXd itSigma(N,N);
    Eigen::VectorXd mu(N);
    Eigen::ArrayXd a0(N);
    Eigen::MatrixXd delta_pw = Eigen::MatrixXd::Zero(2*N,N);
    Eigen::MatrixXd delta_lg = Eigen::MatrixXd::Zero(2*N,N);
    std::stringstream ss;
    // setting values
    a0 << 1.0, 1.0;
    const double T = 0.45;          // maturity
    const double r = 0.0;           // interest
    Md << 0.00, 0.95, 0.95, 0.00;
    Ms << 0.00, 0.0, 0.0, 0.00;
    M << Ms, Md;
    debt << 11.3, 11.3;
    mu << 10.01*sigma(0,0)*sigma(0,0) , 10.01*sigma(1,1)*sigma(1,1);
    sigma << T, 0., 0., T;              //std::log(1.0*1.0 + 1.0), std::log(1.0*1.0 + 1.0);
    itSigma = (T*sigma).inverse();
    double sig[N][N] = { { sigma(0,0), 0.0},{0., sigma(1,1)} };   // Variance
    trng::correlated_normal_dist<> Z(&sig[0][0], &sig[N-1][N-1]+1);
    trng::lognormal_dist<> v1(mu(0), sigma(0,0));
    trng::lognormal_dist<> v2(mu(1), sigma(1,1)); 


    // running simulation
    for(unsigned int i = 0; i < nPoints; i++)
    {
        //V << mu(0)*T-sigma(0)+Z(gen_v1), mu(1)*T-sigma(1)+Z(gen_v1);
        // --- setup
        Vb << (r - sigma(0,0)*sigma(0,0)/2.)*T + std::sqrt(T)*Z(gen_v1), (r - sigma(1,1)*sigma(1,1)/2.)*T + std::sqrt(T)*Z(gen_v1);
        Vb = Vb.array().exp();
        V = a0*Vb.array();
        auto rs = run_modified(M, V, debt, N, 1000);

        // --- run
        Eigen::VectorXi solvent = classify_solvent(V, debt);

        // --- extract
        Eigen::MatrixXd Jrs_m1 = eye22 - jacobian_rs(M, solvent);
        Eigen::MatrixXd Ja = jacobian_a(solvent);
        //LOG(INFO) << "solvent: \n" << solvent;
        //LOG(INFO) << "Jrs: \n" << Jrs_m1;
        Jrs_m1 = Jrs_m1.inverse();
        //LOG(INFO) << "Jrs_m1: \n" << Jrs_m1;
        //LOG(INFO) << "Ja: \n" << Ja;
        //LOG(INFO) << "Vb: \n" << test;
        //LOG(INFO) << "Jrs_m1*Ja: \n" << Jrs_m1*Ja;
        //LOG(INFO) << std::exp(-r*T)*(Jrs_m1*Ja)*Vb.asDiagonal();
        delta_pw = delta_pw - std::exp(-r*T)*(Jrs_m1*Ja)*(Vb.asDiagonal());
        LOG(INFO) << (V.array().log() - a0.log() - (r-sigma(0)*sigma(0)/2.)*T);
        //delta_lg = delta_lg +  v *std::exp(r*T) * (itSigma * ((V.array().log() - a0.log() - (r-sigma(0)*sigma(0)/2.)*T)/a0).matrix());
        // --- prepare output
        Eigen::VectorXd v_res(N);
        V_out = (eye2 - M)*rs;
        //exit(0);
        for(unsigned int j = 0; j < N; j++)
        {
            v_res(j) = rs(j) + rs(j+N);
            ss << v_res(j) << "\t";
        }
        if(i == 5) exit(0);
        ss << static_cast<std::underlying_type<ACase>::type>(classify(v_res, debt)) << std::endl;
        LOG(INFO) << delta_pw;
        //LOG(INFO) << delta_lg;
    }
    std::cout << ss.str();
}
