#include "BlackScholesNetwork.hpp"

size_t BlackScholesNetwork::gbl_dbg_counter = 0;

BlackScholesNetwork::BlackScholesNetwork(const Eigen::MatrixXd& M, const double T,const double r):
    M(M), T(T), r(r), exprt(std::exp(-r*T)), dbg_counter(gbl_dbg_counter)
{
    gbl_dbg_counter += 1;
    LOG(TRACE) << "creating new BSN object";
    EXPECT_EQ(M.cols(), 2*M.rows()) << "Dimensions for cross holding matrix invalid!";
}


BlackScholesNetwork::BlackScholesNetwork(Eigen::MatrixXd& M, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r):
M(M), S0(S0), St(assets), debt(debt), T(T), r(r), exprt(std::exp(-r*T)), dbg_counter(gbl_dbg_counter)
{
    gbl_dbg_counter += 1;
    EXPECT_EQ(M.cols(), 2*M.rows()) << "Dimensions for cross holding matrix invalid!";
    EXPECT_EQ(assets.rows(), debt.rows()) <<  "Dimensions of debts and asset vector do not match!";
    EXPECT_EQ(assets.rows(), M.rows()) << "Dimensions for assets vector and cross holding matrix to not match!";
};


/** 
 *  which_to_set = 0 (both) 1 (M_s = 0) 2 (M_d = 0)
 */
BlackScholesNetwork::BlackScholesNetwork(double p, double val, char which_to_set, Eigen::VectorXd& S0, Eigen::VectorXd& assets, Eigen::VectorXd& debt, double T, double r):
    S0(S0), St(assets), debt(debt), T(T), r(r), exprt(std::exp(-r*T)), dbg_counter(gbl_dbg_counter)
{
    gbl_dbg_counter += 1;
    auto N = St.size();
    M = Eigen::MatrixXd::Zero(N,2*N);
    EXPECT_GT(p, 0) << "p is not a probability";
    EXPECT_LT(p, 1) << "p is not a probability";
    EXPECT_GT(val, 0) << "val is not a probability";
    EXPECT_LT(val, 1) << "val is not a probability";
    set_M_ER(p, val, which_to_set);
}


void BlackScholesNetwork::set_solvent()
{
    auto N = M.rows();
    solvent.resize(N);
    for(unsigned int i = 0; i < N; i++)
    {
        solvent(i) = 1*(x(i)+x(i+N) >= debt(i));
    }
    LOG(INFO) << "solvent: " << solvent;
}


std::vector<double> BlackScholesNetwork::run_valuation(unsigned int iterations)
{
    LOG(INFO) << "running valuation, object id = " << dbg_counter;
    auto N = M.rows();
    x = Eigen::VectorXd::Zero(2*N);
    Eigen::VectorXd a = S0.array()*St.array();
    double dist = 99.;
    for(unsigned int r = 0; r < iterations; r++)
    {
        auto tmp = a + M*x;
        auto distV = x;
        x.head(N) = (tmp - debt).cwiseMax(0.);
        x.tail(N) = tmp.cwiseMin(debt);
        distV = distV - x; 
        dist = distV.norm();
        if(dist < 1.0e-14)
             break;
    }
    set_solvent();
    return get_rs();
}


void BlackScholesNetwork::set_M_ER(const double p, const double val, char which_to_set)
{
    EXPECT_GT(val, 0) << "val is not a probability";
    EXPECT_LT(val, 1) << "val is not a probability";
    auto N = St.size();
    M = Eigen::MatrixXd::Zero(N, 2*N);
    trng::yarn2 gen_u;
    trng::uniform01_dist<> u_dist;
    //@TODO: use bin. dist. to generate vectorized
    for(int i = 0; i < N;i++)
    {
        for(int j = i+1; j < N; j++)
        {
            if(which_to_set == 1 || which_to_set == 0)
            {
                if(u_dist(gen_u) < p)
                    M(i,j) = 1.0;
                if(u_dist(gen_u) < p)
                    M(j,i) = 1.0;
            }
            if(which_to_set == 2 || which_to_set == 0)
            {
                if(u_dist(gen_u) < p)
                    M(i,j+N) = 1.0;
                if(u_dist(gen_u) < p)
                    M(j,i+N) = 1.0;
            }
        }
    }
    auto col_sum = M.colwise().sum();
    auto row_sum = M.rowwise().sum();
    double max = std::max(col_sum.maxCoeff(), row_sum.maxCoeff());
    M = (val/max)*M;
}


Eigen::MatrixXd BlackScholesNetwork::iJacobian_fx()
{
    const auto N = M.rows();
    Eigen::MatrixXd J(2*N, 2*N);
    //@TODO: replace this loop with stacked solvent matrix x matrix operation?
    //J.topRows(N) = M.array().colwise() * solvent.cast<double>().array();
    //J.bottomRows(N) = M.array().colwise() * (1-solvent.cast<double>().array());

    if(solvent.size() == M.rows())
    {
        for(unsigned int i = 0; i < N; i++)
        {
            J.row(i) = solvent(i)*M.row(i);
            J.row(N+i) = (1-solvent(i))*M.row(i);
        }
        J = (Eigen::MatrixXd::Identity(2*N, 2*N) - J).inverse();
    } else {
        LOG(WARNING) << "solvent has wrong size: " << solvent.size() << "\nobject id = " << dbg_counter;
        J = Eigen::MatrixXd::Zero(2*N,2*N);
    }
    return J;
}


Eigen::MatrixXd BlackScholesNetwork::Jacobian_va()
{
    const auto N = M.rows();
    Eigen::MatrixXd J(2*N, N);
    if(solvent.size() == M.rows())
    {
        Eigen::MatrixXd sd = solvent.cast<double>().asDiagonal();
        J << sd, sd;
    } else {
        LOG(WARNING) << "solvent has wrong size: " << solvent.size();
        J = Eigen::MatrixXd::Zero(2*N,N);
    }
    return J;
}


std::vector<double>  BlackScholesNetwork::get_assets()
{
    std::vector<double> res;
    res.resize(St.size());
    Eigen::VectorXd::Map(&res[0], St.size()) = S0.array()*St.array();
    return res;
}
