

Eigen::MatrixXd run_valuation(Eigen::MatrixXd& vij, Eigen::MatrixXd& zij, Eigen::VectorXd& B, const unsigned int N, const unsigned int max_it, const unsigned int L);
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
