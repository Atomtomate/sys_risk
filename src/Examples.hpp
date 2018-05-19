//
// Created by julian on 5/9/18.
//

#ifndef VALUATION_EXAMPLES_HPP
#define VALUATION_EXAMPLES_HPP

#include "Sampler.hpp"
#include "BlackScholesNetwork.hpp"



class N2_network
{
private:
    const unsigned N = 2;
    const double T = 1.0;                                   // maturity
    const double r = 0.0;                                   // interest
    double sig[2][2];
    BlackScholesNetwork bsn;
    Eigen::VectorXd S0;
    Eigen::VectorXd debt;                            // debt
    Eigen::MatrixXd M;                               // Ms, Md
    Eigen::MatrixXd itSigma;

    trng::yarn2 gen_z;
    Eigen::VectorXd var_h;
    trng::correlated_normal_dist<> Z_dist;

    // last result, returned by observers
    std::vector<double> rs;


    void init_network()
    {
        S0 = Eigen::VectorXd(N);
        debt = Eigen::VectorXd(N);
        M = Eigen::MatrixXd(N, 2*N);
        itSigma = Eigen::MatrixXd(N,N);
        var_h = Eigen::VectorXd(N);
        Eigen::MatrixXd Ms(N,N);                                // cross holdings
        Ms << 0.00, 0.00, 0.00, 0.00;
        Eigen::MatrixXd Md(N,N);                                // cross debt
        Md << 0.00, 0.95, 0.95, 0.00;
        M << Ms, Md;
        S0 << 1.0, 1.0;
        debt << 11.3, 11.3;
        bsn.set_M(M);
        bsn.set_S0(S0);
        bsn.set_debt(debt);

        // random asset stuff
        Eigen::MatrixXd sigma(N,N);
        sigma << T, 0., 0., T;              //std::log(1.0*1.0 + 1.0), std::log(1.0*1.0 + 1.0);
        itSigma = (T*sigma).inverse();
        var_h = T*r - T*sigma.diagonal().array()*sigma.diagonal().array()/2.;
    }


public:
    N2_network():
            bsn(Eigen::MatrixXd::Zero(N,2*N), T, r), sig {{T, 0.}, {0., T}} , Z_dist(&sig[0][0], &sig[N-1][N-1]+1)
    {
        init_network();
    }

    void test_N2_valuation();

    auto draw_from_dist()
    {
        const size_t N = M.rows();
        Eigen::VectorXd S_log(N);             // log of lognormal distribution exogenous assets, without a_0
        Eigen::VectorXd Z(N);                 // Multivariate normal, used to generate lognormal assets
        for(unsigned int d = 0; d < N; d++)
        {
            Z(d) = Z_dist(gen_z);
            S_log(d) = var_h(d) + std::sqrt(T)*Z(d);
        }
        std::vector<double> res;
        res.resize(S_log.size());
        Eigen::VectorXd::Map(&res[0], S_log.size()) = S_log.array().exp();
        return res;
    }

    auto run(std::vector<double> St_in)//Eigen::VectorXd St)
    {
        Eigen::VectorXd St = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(St_in.data(), St_in.size() );

        bsn.set_St(St);
        rs = bsn.run_valuation(1000);
    }

    auto test_out()
    {
        auto v_o = bsn.get_valuation();
        auto s_o = bsn.get_solvent();
        int out_i = 3;
        if(s_o[0] > 0 && s_o[1] > 0) out_i = 0;
        if(s_o[0] > 0 && s_o[1] == 0) out_i = 2;
        if(s_o[0] == 0 && s_o[1] > 0) out_i = 1;
        std::cout << v_o[0] << "\t" << v_o[1] << "\t" << out_i << std::endl;
        std::vector<double> out {0.};
        return out;
    }
};


#endif //VALUATION_EXAMPLES_HPP
