#include "MultiVar_BlackScholes.hpp"

MultiVar_BlackScholes::MultiVar_BlackScholes(Eigen::MatrixXd M, Eigen::VectorXd assets, Eigen::VectorXd deb, double T, double r):
    M(M), assets(assets), debt(debt)
    {
        if(M.cols() != 2*M.rows())
        {
            LOG(WARNING) << "Dimensions for cross holding matrix invalid!";
            throw std::logic_error("Dimensions for cross holding matrix invalid!");
        }
        if(assets.rows() != debt.rows())
        {
            LOG(WARNING) << "Dimensions of debts and asset vector do not match!";
            throw std::logic_error("Dimensions of debts and asset vector do not match!");
        }
        if(assets.rows() != M.rows())
        {
            LOG(WARNING) << "Dimensions for assets vector and cross holding matrix to not match!";
            throw std::logic_error("Dimensions for assets vector and cross holding matrix to not match!");
        }
    };

/** 
 *  which_to_set = 0 (both) 1 (M_s = 0) 2 (M_d = 0)
 */
MultiVar_BlackScholes::MultiVar_BlackScholes(double p, double val, char which_to_set, Eigen::VectorXd assets, Eigen::VectorXd debt)
{
    auto N = assets.size();
    M = Eigen::MatrixXd::Zero(N,2*N);
    if(p > 1 || p < 0 || val > 1 || val < 0)
        throw std::logic_error("Cross holding probability or value not in valid range");
    set_M_ER(p, val, which_to_set);
}


void MultiVar_BlackScholes::set_solvent(void)
{
    auto N = assets.size();
    Eigen::VectorXi solvent = Eigen::VectorXi::Zero(N);
    for(unsigned int i = 0; i < N; i++)
    {
        solvent(i) = 1*(x(i)+x(i+N) >= debt(i));
    }
}

Eigen::VectorXd MultiVar_BlackScholes::run_valuation(unsigned int iterations)
{
    auto N = assets.size();
    x = Eigen::VectorXd::Zero(2*N);
    double dist = 99.;
    for(unsigned int r = 0; r < iterations; r++)
    {
        auto tmp = assets + M*x;
        auto distV = x;
        x.head(N) = (tmp - debt).cwiseMax(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        distV = distV - x; 
        dist = distV.norm();
        if(dist < 1.0e-14)
             break;
    }
    set_solvent();
    return x;
}

void MultiVar_BlackScholes::set_M_ER(const double p, const double val, char which_to_set)
{ 
    auto N = assets.size();
    M = Eigen::MatrixXd::Zero(N, 2*N);
    trng::yarn2 gen_u;
    trng::uniform_dist<> u_dist(0, 1);
    for(int i = 0; i < N;i++)
    {
        for(int j = i+1; j < N; j++)
        {
            if(which_to_set == 1 || which_to_set == 0)
            {
                if(u_dist(gen_u) < p)
                    M(i,j) = val;
                if(u_dist(gen_u) < p)
                    M(j,i) = val;
            }
            if(which_to_set == 2 || which_to_set == 0)
            {
                if(u_dist(gen_u) < p)
                    M(i,j+N) = val;
                if(u_dist(gen_u) < p)
                    M(j,i+N) = val;
            }
        }
    }
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


void run_greeks(void)
{
    trng::yarn2 gen_v1;             // prng
    const unsigned N = 2;           // number of firms
    const unsigned nPoints = 10000; // number of data points
    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(N,N);
    Eigen::MatrixXd eye2(N,2*N);
    Eigen::MatrixXd eye22 = Eigen::MatrixXd::Identity(2*N, 2*N);
    eye2 << eye, eye;
    Eigen::MatrixXd Md(N,N);                                // cross debt
    Eigen::MatrixXd Ms(N,N);                                // cross holdings
    Eigen::MatrixXd M(N,2*N);                               // Ms, Md
    Eigen::VectorXd V_out = Eigen::VectorXd(N);             // outside investors
    Eigen::VectorXd debt(N);                                // debt
    Eigen::MatrixXd sigma(N,N);                             
    Eigen::MatrixXd itSigma(N,N);
    Eigen::VectorXd v_res = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd rs_acc = Eigen::VectorXd::Zero(2*N);
    Eigen::VectorXd mu = Eigen::VectorXd(N);
    Eigen::VectorXd S0 = Eigen::VectorXd(N);

    Eigen::MatrixXd delta_pw = Eigen::MatrixXd::Zero(2*N,N);
    Eigen::MatrixXd delta_lg = Eigen::MatrixXd::Zero(2*N,N);
    std::stringstream ss;

    // setting values for example calculation
        S0 << 1.0, 1.0;
        const double T = 1.0;          // maturity
        const double r = 0.0;           // interest
        Ms << 0.00, 0.0, 0.0, 0.00;
        Md << 0.00, 0.95, 0.80, 0.00;
        M << Ms, Md;
        debt << 11.3, 11.3;
        mu << -0.5*sigma(0,0)*sigma(0,0) + std::log(S0(0)) , -0.5*sigma(1,1)*sigma(1,1) + std::log(S0(1));
        sigma << T, 0., 0., T;              //std::log(1.0*1.0 + 1.0), std::log(1.0*1.0 + 1.0);
        itSigma = (T*sigma).inverse();
        double sig[N][N] = { { sigma(0,0), sigma(0,1)},{sigma(1,0), sigma(1,1)} };   // Variance
        trng::correlated_normal_dist<> Z_dist(&sig[0][0], &sig[N-1][N-1]+1);
        Eigen::VectorXd var_h = T*r - T*sigma.diagonal().array()*sigma.diagonal().array()/2;

    // running simulation
    for(unsigned int i = 0; i < nPoints; i++)
    {
        Eigen::VectorXd S_log = Eigen::VectorXd(N);             // log of lognormal distr exogeneous assets, without a_0
        Eigen::VectorXd Z = Eigen::VectorXd(N);                 // Multivariate normal, used to generate lognormal assets
        // --- setup
        for(unsigned int d = 0; d < N; d++)
        {
            Z(d) = Z_dist(gen_v1);
            S_log(d) = var_h(d) + std::sqrt(T)*Z(d);
        }
        Eigen::VectorXd St = S_log.array().exp();               // assets without initial value a_0
        auto rs = run_modified(M, S0.array()*St.array(), debt, N, 1000);

        // --- run
        v_res = rs.head(N) + rs.tail(N);
        Eigen::VectorXi solvent = classify_solvent(v_res, debt);

        // --- extract
        Eigen::MatrixXd Jrs_m1 = (eye22 - jacobian_rs(M, solvent)).inverse();
        Eigen::MatrixXd Ja = jacobian_a(solvent);
        //Eigen::VectorXd tmp = (S0.array()*St.array()).log() - S0.array().log() - var_h.array();
        Eigen::VectorXd ln_fac = (itSigma*Z).array()/S0.array();
        delta_pw = delta_pw + std::exp(-r*T)*(Jrs_m1*Ja)*(St.asDiagonal());
        delta_lg = delta_lg + std::exp(-r*T)*(ln_fac*(rs.transpose())).transpose();
        /*
        LOG(INFO) << "====== DEBUG OUTPUT FOR ITERATION " << i << " ======";
        LOG(INFO) << "St: \n" << St;
        LOG(INFO) << "debt: \n" << debt;
        LOG(INFO) << "Solvent: \n" << solvent;
        LOG(INFO) << "Jrs_m1: \n" << Jrs_m1;
        LOG(INFO) << "Ja: \n" << Ja;
        LOG(INFO) << "S_w0: \n" << S_w0;
        LOG(INFO) << "Jrs_m1*Ja: \n" << Jrs_m1*Ja;
        LOG(INFO) << "Delta_J: \n" << (Jrs_m1*Ja)*(S_w0.asDiagonal());
        LOG(INFO) << "(t Sigma)^{-1}: \n" << itSigma;
        LOG(INFO) << "z: \n" << (z_v);
        LOG(INFO) << "itSigma*z_v: \n" << itSigma*z_v;
        LOG(INFO) << "rs: \n" << rs;
        LOG(INFO) << "Delta_L: \n" << (ln_fac*(v_res.transpose())).transpose();
        LOG(INFO) << "-----------------------------------------------------";
        //LOG(DEBUG) << (itSigma*z_v).array();
        //LOG(DEBUG) << S0.array();
        //LOG(DEBUG) << (itSigma*z_v).array() / S0.array();
        if(i > 2)
            exit(0);
       */
        rs_acc += rs;

        // prepare output
        V_out = (eye2 - M)*rs;
        for(unsigned int j = 0; j < N; j++)
        {
            ss << v_res(j) << "\t";
        }
        ss << static_cast<std::underlying_type<ACase>::type>(classify(v_res, debt)) << std::endl;
    }
    std::cout << ss.str();
    rs_acc = rs_acc/nPoints;
    v_res = rs_acc.head(N) + rs_acc.tail(N);
    /*LOG(INFO) << "delta_pw: \n" << delta_pw/nPoints;
    LOG(INFO) << "delta_lg: \n" << delta_lg/nPoints;
    LOG(INFO) << "rs: \n" << rs_acc;
    LOG(INFO) << "v: \n" << v_res;*/
}
