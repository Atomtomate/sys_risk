/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

/*Eigen::Matrix<stan::math::var, Dynamic, 1> test_stan_math_f(Eigen::Matrix<stan::math::var, Dynamic, 1> x_var)
{
    Eigen::Matrix<stan::math::var, Dynamic, 1> res = x_var; //(2*x_var.size());
    res << x_var(1), x_var(0);
    return res;
}

void test_stan_math() {
    Eigen::VectorXd x(2);
    x << 1.0,2.0;
    Eigen::Matrix<stan::math::var, Dynamic, 1> x_var(x.size());
    for (int i = 0; i < x.size(); ++i)
        x_var(i) = x(i);
    Eigen::Matrix<stan::math::var, Dynamic, 1> f_x_var = test_stan_math_f(x_var);
    Eigen::VectorXd f_x(f_x_var.size());
    for (int i = 0; i < f_x.size(); ++i)
        f_x(i) = f_x_var(i).val();
    Eigen::MatrixXd J(f_x_var.size(), x_var.size());
    for (int i = 0; i < f_x_var.size(); ++i) {
        if (i > 0)
            stan::math::set_zero_all_adjoints();
        f_x_var(i).grad();
        for (int j = 0; j < x_var.size(); ++j)
            J(i,j) = x_var(j).adj();
    }
    LOG(INFO) << "x: \n" << x;
    LOG(INFO) << "f(x): \n" << f_x;
    LOG(INFO) << "J: \n" << J;
}
*/
