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

namespace py = pybind11;



PYBIND11_MODULE(PyVal, m) {

    py::class_<Py_ER_Net>(m, "BS_Network")
            .def(py::init<>())
            .def("run", &Py_ER_Net::run_valuation)
            .def("get_M", &Py_ER_Net::get_M)  //, py::return_value_policy::reference_internal)
            .def("get_rs", &Py_ER_Net::get_rs) // , py::return_value_policy::reference_internal)
            .def("get_solvent", &Py_ER_Net::get_solvent) //, py::return_value_policy::reference_internal);
            .def("get_assets", &Py_ER_Net::get_assets) //, py::return_value_policy::reference_internal);
            .def("get_delta_jacobians", &Py_ER_Net::get_delta_jac) //, py::return_value_policy::reference_internal);
            .def("get_rs_var", &Py_ER_Net::get_rs_var) // , py::return_value_policy::reference_internal)
            .def("get_solvent_var", &Py_ER_Net::get_solvent_var) //, py::return_value_policy::reference_internal);
            .def("get_assets_var", &Py_ER_Net::get_assets_var) //, py::return_value_policy::reference_internal);
            .def("get_delta_jacobians_var", &Py_ER_Net::get_delta_jac_var); //, py::return_value_policy::reference_internal);

}

#endif //VALUATION_PYTHONINTERFACE_HPP
