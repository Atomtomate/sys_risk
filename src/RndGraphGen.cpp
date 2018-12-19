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
    constexpr bool allow_unnormalized_rows = true;
    constexpr bool check_suppression = false;
    constexpr bool check_selfloop = true;
    constexpr bool check_multiedge = true;
    constexpr int max_it = 3000;
    constexpr int max_rej_it = 10000;

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


    void gen_sinkhorn(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const double p, const double val,
                 const int which_to_set) {
        const int N = M->rows();
        trng::uniform01_dist<> u_dist;
        int rej_it = 0;
        const double smallest_el = val/static_cast<double>(N);
        bool cols_ok = false;
        if(which_to_set != 2) LOG(ERROR) << "mode 0 and 1 for M generation not yet implemented!";
        Eigen::MatrixXd M_bak = (*M);
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
            Eigen::VectorXd col_sums = M->rightCols(N).colwise().sum();
            do {
                cols_ok = true;
                //M_bak = (*M);
                //        LOG(ERROR) << (*M);
                for (int ii = 0; ii < N; ii++) {
                    if (col_sums(ii) > 0)
                        M->rightCols(N).col(ii) = (val / col_sums(ii)) * M->rightCols(N).col(ii);
                }
                //LOG(INFO) << (*M);

                if constexpr (!allow_unnormalized_rows) {
                    Eigen::VectorXd row_sums = M->rightCols(N).rowwise().sum();
                    col_sums =  M->rightCols(N).colwise().sum();
                    for (int ii = 0; ii < row_sums.size(); ii++) {
                        if (row_sums(ii) > 0)
                            M->rightCols(N).row(ii) = (val / row_sums(ii)) * M->rightCols(N).row(ii);
                    }


                    for (int ii = 0; (ii < row_sums.size()) && cols_ok; ii++) {
                        //if(row_sums(ii) >= 1.0)
                        //    cols_ok = false;
                        if ((row_sums(ii) > eps) && (std::abs(row_sums(ii) - val) > eps))
                            cols_ok = false;
                    }
                    if(!cols_ok && M->isApprox(M_bak))      // oscillating
                    {
                        cols_ok = false;
                        it = max_it;
                    }

                    for(int ii = 0;it < max_it && ii < N; ii++)
                    {
                        for(int jj = 0 ; jj < N; jj++)
                        {
                            if((*M)(ii,jj+N) > 0 && (*M)(ii,jj+N) < smallest_el) {
                                if constexpr (check_suppression){
                                    cols_ok = false;
                                    it = max_it;
                                } else {
                                    (*M)(ii,jj+N) = 0;
                                }
                                //LOG(WARNING) << "found suppressed element at (" << ii << ", " << jj << ")";
                            }
                        }
                    }
                    it++;
                }

            } while (!cols_ok && it < max_it);
            rej_it++;
        } while (!cols_ok && rej_it < max_rej_it);
        if (rej_it > max_rej_it - 1){
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
            const int N = M->rows();
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


    void gen_fixed_degree(Eigen::MatrixXd *M, trng::yarn2 &gen_u, const double p, const double val,
                          const int which_to_set) {
        int degree = std::floor(p * M->rows());
        if (which_to_set == 1 || which_to_set == 0) {
            gen_fixed_degree_internal(M, gen_u, degree, val, which_to_set, 0);
        }
        if (which_to_set == 2 || which_to_set == 0) {
            gen_fixed_degree_internal(M, gen_u, degree, val, which_to_set, M->rows());
        }
    }


    void gen_configuration_model(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val,
                                 const int which_to_set)
    {
        const int N = M->rows();
        const int degree = std::floor(p*N);
        const double smallest_el = val/static_cast<double>(N);
        int rej_it = 0;
        bool cols_ok = false;
        if(which_to_set != 2) LOG(ERROR) << "mode 0 and 1 for M generation not yet implemented!";

        do {
            int it = 0;
            M->setZero();
            M->setZero();
            std::vector<int> in(N*degree);
            for(int i = 0; i < N; i++)
                std::fill(in.begin()+degree*i, in.begin()+degree*(i+1), i);
            std::random_shuffle(in.begin(), in.end());
            for(int i = 0; i < N*degree; i++)
            {
                const int row = (int)(i/degree);
                const int col = in[i];
                if(row == col)
                    continue;
                (*M)(row, col + N) += 1;
            }

            Eigen::MatrixXd M_bak = (*M);
            Eigen::VectorXd col_sums = M->rightCols(N).colwise().sum();
            do {
                cols_ok = true;
                M_bak = (*M);
                for (int ii = 0; ii < N; ii++) {
                    if (col_sums(ii) > 0)
                        M->rightCols(N).col(ii) = (val / col_sums(ii)) * M->rightCols(N).col(ii);
                }
                Eigen::VectorXd row_sums = M->rightCols(N).rowwise().sum();
                for (int ii = 0; ii < row_sums.size(); ii++) {
                    if (row_sums(ii) > 0)
                        M->rightCols(N).row(ii) = (val/ row_sums(ii)) * M->rightCols(N).row(ii);
                }

                col_sums = M->rightCols(N).colwise().sum();
                for (int ii = 0; (ii < col_sums.size()) && cols_ok; ii++) {
                    if((col_sums(ii) > eps) && (std::abs(col_sums(ii) - val) > eps))
                        cols_ok = false;
                }

                if(!cols_ok && M->isApprox(M_bak))      // oscillating
                {
                    cols_ok = false;
                    it = max_it;
                }

                for(int ii = 0;it < max_it && ii < N; ii++)
                {
                    for(int jj = 0 ; jj < N; jj++)
                    {
                        if((*M)(ii,jj+N) > 0 && (*M)(ii,jj+N) < smallest_el) {
                            if constexpr (check_suppression){
                                cols_ok = false;
                                it = max_it;
                            } else {
                                (*M)(ii,jj+N) = 0;
                            }
                        }
                    }
                }
                it++;
            } while (!cols_ok && it < max_it);
            rej_it++;
        } while (!cols_ok && rej_it < max_rej_it);
        if ((rej_it > max_rej_it - 1)) {
            throw std::runtime_error(
                    "\n\nToo many rejections during generation of M! p=" + std::to_string(p) + ", n=" +
                    std::to_string(N) + "\n\n");
        }
    }


    Eigen::MatrixXd in_out_degree(Eigen::MatrixXd *M) {
        Eigen::MatrixXd in_out_deg = Eigen::MatrixXd::Zero(2, M->cols());
        for (int i = 0; i < M->rows(); i++) {
            int in_deg = 0;
            for (int j = 0; j < M->cols(); j++) {
                in_deg += (int) ((*M)(i, j) > eps);
            }
            in_out_deg(0, i) += in_deg;
        }
        for (int i = 0; i < M->cols(); i++) {
            int out_deg = 0;
            for (int j = 0; j < M->rows(); j++) {
                out_deg += (int) ((*M)(j, i) > eps);
            }
            in_out_deg(1, i) += out_deg;

        }
        return in_out_deg;
    }

    int fixed_degree(Eigen::MatrixXd* M)
    {
        int res = -1;
        int N = M->rows();
        for (int i = 0; i < M->rows(); i++) {
            int in_deg = 0;
            for (int j = 0; j < M->leftCols(N).cols(); j++) {
                in_deg += (int) ((*M)(i, j + N) > eps);
            }
            if(res == -1)
                res = in_deg;
            else if(res != in_deg)
                return -1;
        }
        for (int i = 0; i < M->leftCols(N).cols(); i++) {
            int out_deg = 0;
            for (int j = 0; j < M->rows(); j++) {
                out_deg += (int) ((*M)(i, j + N) > eps);
            }
            if(res != out_deg)
                return -1;
        }
        return res;
    }

    Eigen::MatrixXd avg_row_col_sums(Eigen::MatrixXd* M)
    {
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(2, M->cols());
        res.leftCols(M->rows()).topRows(1) = M->rowwise().sum().transpose();
        res.bottomRows(1).array() = M->colwise().sum().array();
        return res;
    }

    std::pair<double,double> avg_io_deg(Eigen::MatrixXd* M)
    {
        std::pair<double, double> res(0., 0.);
        int N = M->rows();
        for (int i = 0; i < N; i++) {
            int in_deg = 0;
            int out_deg = 0;
            for (int j = 0; j < N; j++) {
                in_deg += (int) ((*M)(j, i + N) > eps);
                out_deg += (int) ((*M)(i, j + N) > eps);
            }
            res.first += in_deg;
            res.second += out_deg;
        }
        res.first = res.first/N;
        res.second = res.second/N;
        return res;
    }


    void fixed_2d(Eigen::MatrixXd* M, const double vs01, const double vs10, const double vr01, const double vr10)
    {
        M->setZero();
        (*M)(0, 3) = vr01;
        (*M)(1, 2) = vr10;
        (*M)(0, 1) = vs01;
        (*M)(1, 0) = vs10;
    }


    void gen_ring(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set)
    {
        M->setZero();
        auto N = M->rows();
        double v = val/(N-1);
        if( which_to_set == 1 || which_to_set == 0)
        {
            M->diagonal(1) = Eigen::VectorXd::Constant(M->rows(), v);
            (*M)(N-1,0) = v;
        }
        if(which_to_set == 2 || which_to_set == 0)
        {
            M->diagonal(M->rows()+1) = Eigen::VectorXd::Constant(M->rows(), v);
            (*M)(N-1,N) = v;
        }
    }

    void gen_star(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set)
    {
        M->setZero();
        auto N = M->rows();
        double v = val/(N-1);
        if( which_to_set == 1 || which_to_set == 0)
        {
            M->col(0) = Eigen::VectorXd::Constant(N, v);
            M->leftCols(N).row(0) = Eigen::VectorXd::Constant(N, v);
            (*M)(0,0) = 0;
        }
        if(which_to_set == 2 || which_to_set == 0)
        {
            M->col(N) = Eigen::VectorXd::Constant(N, v);
            M->rightCols(N).row(0) = Eigen::VectorXd::Constant(N, v);
            (*M)(0,N) = 0;
        }
    }

    void gen_uniform(Eigen::MatrixXd* M, trng::yarn2& gen_u, const double p, const double val, const int which_to_set)
    {
        M->setZero();
        auto N = M->rows();
        double v = val/(N-1);
        if( which_to_set == 1 || which_to_set == 0)
        {
            M->leftCols(N) = Eigen::VectorXd::Constant(N,N, v);
            M->diagonal() = Eigen::VectorXd::Constant(M->rows(), 0.);
        }
        if(which_to_set == 2 || which_to_set == 0)
        {
            M->rightCols(N) = Eigen::VectorXd::Constant(N,N, v);
            M->diagonal(N) = Eigen::VectorXd::Constant(M->rows(), 0.);
        }
    }

}
