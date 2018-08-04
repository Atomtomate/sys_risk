/* Copyright (C) 7/30/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#include "RndGraphGen.hpp"

namespace Utils {
    constexpr double eps = 1e-5;

    void gen_basic_rejection(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const double p,
                             const double val, const int which_to_set) {
        const int N = M->rows();
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


    void
    gen_sinkhorn(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const double p, const double val_row, const double val_col,
                 const int which_to_set) {
        const int N = M->rows();
        trng::uniform01_dist<> u_dist;
        int rej_it = 0;
        bool cols_ok = false;
        do {
            int it = 0;
            M->setZero();

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
            Eigen::VectorXd col_sums = M->colwise().sum();
            do {
                cols_ok = true;

                for (int ii = 0; ii < 2 * N; ii++) {
                    if (col_sums(ii) > 0)
                        M->col(ii) = (val_col / col_sums(ii)) * M->col(ii);
                }
                Eigen::VectorXd row_sums = M->rowwise().sum();
                for (int ii = 0; ii < N; ii++) {
                    if (row_sums(ii) > 0)
                        M->row(ii) = (val_row / row_sums(ii)) * M->row(ii);
                }

                col_sums = M->colwise().sum();
                for (int ii = 0; (ii < 2 * N) && cols_ok; ii++) {
                    if(col_sums(ii) > eps && (std::abs(col_sums(ii) - val_col) > eps))
                        cols_ok = false;
                }
                it++;
            } while (!cols_ok && it < 1000);
            rej_it++;
        } while (!cols_ok && rej_it < 1000);
        if (rej_it > 999) {
            LOG(ERROR) << (*M);
            LOG(ERROR) << M->colwise().sum();
            LOG(ERROR) << M->rowwise().sum();
            throw std::runtime_error(
                    "\n\nToo many rejections during generation of M! p=" + std::to_string(p) + ", n=" +
                    std::to_string(N) + "\n\n");
        }
    }

    // TODO: in and out degree
    void gen_fixed_degree_internal(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const int degree, const double val,
                                   const int which_to_set, int offs) {
        bool gen_ok = true;
        bool force_on_diag = false;
        int rej_it = 0;
        do {
            gen_ok = true;
            force_on_diag = false;
            int N = M->rows();
            if (N < degree) throw std::logic_error("Degree larger than size of network");
            trng::uniform01_dist<> u_dist;
            M->setZero();
            std::vector<int> possible_indices(N);       // not yet filled
            std::vector<int> forced_indices(N, degree); // must be filled <=> filled all zeros: if arr[i] <=0, M(ri,i) must be set to val
            for (int ii = 0; ii < N; ii++)
                possible_indices[ii] = ii;              // all indeces possible at the start
            std::vector<int> used_indices(N, degree);   // every index can be used >degree< times

            for (int ri = 0; ri < N; ri++) {
                int n_forced = 0;
                // insert forced elements
                for(int jj = 0; jj < N; jj++)
                {
                    if( ((N-ri) - used_indices[jj] < 1) )                           // remaining rows minus number of remaining positions
                    {
                        if(jj == ri)
                        {
                            force_on_diag = true;
                        }
                        (*M)(ri, jj + offs) = val / static_cast<double>(degree);
                        for(unsigned long k = 0; k < possible_indices.size(); k++)
                        {
                            if(possible_indices[k] == jj)
                                possible_indices.erase(possible_indices.begin() + k);
                        }
                        n_forced += 1;
                    }
                }
                std::vector<int> possible_indices_copy(possible_indices);
                for (unsigned long jj = 0; jj < possible_indices_copy.size(); jj++) {
                    if (possible_indices_copy[jj] == ri)
                        possible_indices_copy.erase(possible_indices_copy.begin() + jj);
                }

                // distribute the rest randomly
                for (int i = 0; i < degree - n_forced; i++) {
                    if (possible_indices_copy.size() == 0)                                      // continue if no more elements can be placed
                        continue;

                    // randomly select index under possible indices and set to val
                    int sub_index = std::floor(u_dist(gen_u) * (possible_indices_copy.size()));
                    int index = possible_indices_copy[sub_index];
                    (*M)(ri, index + offs) = val / static_cast<double>(degree);

                    // book keeping
                    if (used_indices[index] == 1) {
                        for (unsigned long jj = 0; jj < possible_indices.size(); jj++) {
                            if (possible_indices[jj] == index)
                                possible_indices.erase(possible_indices.begin() + jj);
                        }
                    }
                    used_indices[index] -= 1;                                                   // upper limit, remove from possible candidates if degree is reached
                    possible_indices_copy.erase(possible_indices_copy.begin() + sub_index);     // in-line book keeping
                }
            }
            auto check = in_out_degree(M);
            gen_ok = true;
            for (int i = 0; i < check.cols(); i++) {
                if ((check(1, i) != 0 || check(0, i) != 0) && i != degree)
                    gen_ok = false;
            }
            rej_it++;
        } while ((!gen_ok || force_on_diag) && rej_it < 10);
        if(rej_it >= 10)
            throw std::runtime_error("Too many rejections");
    }


    void gen_fixed_degree(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const int degree, const double val,
                          const int which_to_set) {
        if (which_to_set == 1 || which_to_set == 0) {
            gen_fixed_degree_internal(M, gen_u, degree, val, which_to_set, 0);
        }
        if (which_to_set == 2 || which_to_set == 0) {
            gen_fixed_degree_internal(M, gen_u, degree, val, which_to_set, M->rows());
        }
    }


    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd *M) {
        Eigen::MatrixXd in_out_deg = Eigen::MatrixXd::Zero(2, M->rows());
        int N = M->rows();
        for (int i = 0; i < M->rows(); i++) {
            int in_deg = 0;
            for (int j = 0; j < M->leftCols(N).cols(); j++) {
                in_deg += (int) ((*M)(i, j + N) > 0);
            }
            in_out_deg(0, in_deg) += 1;
        }
        for (int i = 0; i < M->leftCols(N).cols(); i++) {
            int out_deg = 0;
            for (int j = 0; j < M->rows(); j++) {
                out_deg += (int) ((*M)(i, j + N) > 0);
            }
            in_out_deg(1, out_deg) += 1;
        }
        in_out_deg = in_out_deg/N;
        return in_out_deg;
    }

}
