/* Copyright (C) 5/28/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_PYTHONINTERFACE_HPP
#define VALUATION_PYTHONINTERFACE_HPP

//@TODO: obtain source dircetory from cmake
#include "../src/Py_ER_Net.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

namespace py = pybind11;



PYBIND11_MODULE(PyVal, m) {

    py::class_<Py_ER_Net>(m, "BS_Network")
            .def(py::init<>())
            .def("run", &Py_ER_Net::run_valuation)
            .def("k_vals", &Py_ER_Net::get_k_vals)
            .def("get_N_samples", &Py_ER_Net::get_N_samples)
            .def("get_M", &Py_ER_Net::get_M)  //, py::return_value_policy::reference_internal)
            .def("get_rs", &Py_ER_Net::get_rs) // , py::return_value_policy::reference_internal)
            .def("get_solvent", &Py_ER_Net::get_solvent) //, py::return_value_policy::reference_internal);
            .def("get_assets", &Py_ER_Net::get_assets) //, py::return_value_policy::reference_internal);
            .def("get_delta_jacobians", &Py_ER_Net::get_delta_jac) //, py::return_value_policy::reference_internal);
            .def("get_vega", &Py_ER_Net::get_vega)
            .def("get_vega_var", &Py_ER_Net::get_vega_var)
            .def("get_theta", &Py_ER_Net::get_theta)
            .def("get_theta_var", &Py_ER_Net::get_theta_var)
            .def("get_rho", &Py_ER_Net::get_rho)
            .def("get_rho_var", &Py_ER_Net::get_rho_var)
            .def("get_M_var", &Py_ER_Net::get_M_var)  //, py::return_value_policy::reference_internal)
            .def("get_rs_var", &Py_ER_Net::get_rs_var) // , py::return_value_policy::reference_internal)
            .def("get_solvent_var", &Py_ER_Net::get_solvent_var) //, py::return_value_policy::reference_internal);
            .def("get_assets_var", &Py_ER_Net::get_assets_var) //, py::return_value_policy::reference_internal);
            //.def("get_io_deg_dist", &Py_ER_Net::get_io_deg_dist)
            //.def("get_io_deg_dist_var", &Py_ER_Net::get_io_deg_dist_var)
            .def("get_delta_jacobians_var", &Py_ER_Net::get_delta_jac_var); //, py::return_value_policy::reference_internal);

    py::add_ostream_redirect(m, "ostream_redirect");

}

#endif //VALUATION_PYTHONINTERFACE_HPP
