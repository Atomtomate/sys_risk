#include "Valuation.hpp"

Eigen::MatrixXd run_valuation(Eigen::MatrixXd& vij, Eigen::MatrixXd& zij, Eigen::VectorXd& B, const unsigned int N, const unsigned int max_it, const unsigned int L)
{
    Eigen::VectorXd Z(2*N);
    Z = Eigen::VectorXd::Zero(2*N);
    for(unsigned int li = 0; li < L; li++)
    {
        Eigen::VectorXd Zl(2*N);
        Zl = Eigen::VectorXd::Zero(2*N);
        Eigen::VectorXd V(N);
        //V << 2.0, 0.5, 0.6, 0.6; // TODO: only for testing, generate random vector
        V = Eigen::VectorXd::Random(N);
        for(unsigned int r = 0; r < max_it; r++)
        {
            auto tmp = vij*V + zij*Zl;
            Zl.head(N) = (tmp-B).cwiseMax(0.);
            Zl.tail(N) = tmp.cwiseMin(B);
        }
        Z += Zl;
    }
    Z = Z/L;
    return Z;
}
