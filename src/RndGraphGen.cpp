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
        bool rows_ok = false;
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
                rows_ok = true;

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
                    if ((col_sums(ii) - val_col) > eps)
                        cols_ok = false;
                }
                row_sums = M->rowwise().sum();
                for (int ii = 0; (ii < N) && rows_ok; ii++) {
                    if ((row_sums(ii) - val_row) > eps)
                        rows_ok = false;
                }
                it++;
            } while ((!cols_ok || !rows_ok) && it < 1000);
            rej_it++;
        } while ((!cols_ok || !rows_ok) && rej_it < 1000);
        if (rej_it > 999) {
            LOG(ERROR) << (*M);
            LOG(ERROR) << M->colwise().sum();
            LOG(ERROR) << M->rowwise().sum();
            throw std::runtime_error(
                    "\n\nToo many rejections during generation of M! p=" + std::to_string(p) + ", n=" +
                    std::to_string(N) + "\n\n");
        }
        LOG(ERROR) << rej_it;
    }

    // TODO: in and out degree
    void gen_fixed_degree_internal(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const int degree, const double val,
                                   const int which_to_set, int offs) {
        bool gen_ok = true;
        do {
            int rej_it = 0;
            int N = M->rows();
            if (N < degree) throw std::logic_error("Degree larger than size of network");
            trng::uniform01_dist<> u_dist;
            M->setZero();
            std::vector<int> possible_indices(N);
            for (int ii = 0; ii < N; ii++)
                possible_indices[ii] = ii;
            std::vector<int> used_indices(N, degree);

            for (int ri = 0; ri < N; ri++) {
                std::vector<int> possible_indices_copy(possible_indices);
                for (unsigned long jj = 0; jj < possible_indices_copy.size(); jj++) {
                    if (possible_indices_copy[jj] == ri)
                        possible_indices_copy.erase(possible_indices_copy.begin() + jj);
                }
                for (int i = 0; i < degree; i++) {
                    if (possible_indices_copy.size() == 0)
                        continue;
                    int sub_index = std::floor(u_dist(gen_u) * (possible_indices_copy.size()));
                    int index = possible_indices_copy[sub_index];
                    (*M)(ri, index + offs) = val / static_cast<double>(degree);
                    if (used_indices[index] == 1) {
                        for (unsigned long jj = 0; jj < possible_indices.size(); jj++) {
                            if (possible_indices[jj] == index)
                                possible_indices.erase(possible_indices.begin() + jj);
                        }
                    }
                    used_indices[index] -= 1;
                    possible_indices_copy.erase(possible_indices_copy.begin() + sub_index);
                }
            }
            gen_ok = true;
            auto check = in_out_degree(M);
            for (int i = 0; i < check.cols(); i++) {
                if ((check(1, i) != 0 || check(0, i) != 0) && i != degree)
                    gen_ok = false;
            }
        } while (!gen_ok);
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
