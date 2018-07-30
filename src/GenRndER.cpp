/* Copyright (C) 7/30/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#include "GenRndER.hpp"

namespace Utils {

    void gen_basic_rejection(Eigen::MatrixXd* M, trng::yarn2 gen_u, const double p,
                             const double val, const int which_to_set)
    {
        int N = M->rows();
        int i = 0;
        trng::uniform01_dist<> u_dist;

        do {
            M->setZero();
            //@TODO: use bin. dist. to generate vectorized,
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    if (which_to_set == 1 || which_to_set == 0) {
                        if (u_dist(gen_u) < p)
                            (*M)(i, j) = 1.0;
                        if (u_dist(gen_u) < p)
                            (*M)(j, i) = 1.0;
                    }
                    if (which_to_set == 2 || which_to_set == 0) {
                        if (u_dist(gen_u) < p)
                            (*M)(i, j + N) = 1.0;
                        if (u_dist(gen_u) < p)
                            (*M)(j, i + N) = 1.0;
                    }
                }
            }
            //@TODO: implement S sums
            auto sum_d_j = M->rightCols(N).rowwise().sum();
            auto sum_s_j = M->leftCols(N).rowwise().sum();
            auto sum_i = M->colwise().sum();
            //for(int ii = 0; ii < 2*N; ii++)
            //{
            //    if(sum_i(ii) > 0)
            //       M.col(ii) = M.col(ii)/sum_i(ii);
            //}
            for (int ii = 0; ii < N; ii++) {
                if (sum_d_j(ii) > 0)
                    M->row(ii) = (val / sum_d_j(ii)) * M->row(ii);
            }
            if (i > 100000) {
                throw std::runtime_error(
                        "\n\nToo many rejections during generation of M! p=" + std::to_string(p) + ", n=" +
                        std::to_string(N) + "\n\n");
            }
            i++;
        } while (M->colwise().sum().maxCoeff() > 1);
        if (i > 500) LOG(WARNING) << "\rrejected " << i << " candidates for network matrix";
    }

}
