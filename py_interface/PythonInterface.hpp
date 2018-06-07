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
            .def("add", &Py_ER_Net::add)
            .def("run", &Py_ER_Net::run_valuation);
            //.def("view_M", &Py_ER_Net::view_M, py::return_value_policy::reference_internal)
            //.def("view_rs", &Py_ER_Net::view_rs, py::return_value_policy::reference_internal);

//    m.doc() = "documentation string"; // optional
//    m.def("add", [](int a, int b) { return a + b; });
}

#endif //VALUATION_PYTHONINTERFACE_HPP
