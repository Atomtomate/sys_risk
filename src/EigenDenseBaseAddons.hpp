/* Copyright (C) 5/27/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * adapted from: https://stackoverflow.com/questions/18382457/eigen-and-boostserialize
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_EIGEN_DENSE_BASE_ADDONS_HPP
#define VALUATION_EIGEN_DENSE_BASE_ADDONS_HPP

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/array_wrapper.hpp>

friend class boost::serialization::access;
template<class Archive>
void save(Archive & ar, const unsigned int version) const {
    derived().eval();
    const Index rows = derived().rows(), cols = derived().cols();
    const int size = static_cast<int>(rows*cols);
    ar & boost::serialization::make_nvp("rows", rows);
    ar & boost::serialization::make_nvp("cols", cols);
    // @TODO make_binary_object here?
    ar & boost::serialization::make_nvp("data",(boost::serialization::make_array(derived().data(), size)));//derived().data(), rows*cols)));
    //for (Index j = 0; j < cols; ++j )
    //    for (Index i = 0; i < rows; ++i )
            //ar & tmp[j + i*j];//(derived().coeff(i, j));
}

template<class Archive>
void load(Archive & ar, const unsigned int version) {
    Index rows, cols;
    ar & rows;
    ar & cols;
    if (rows != derived().rows() || cols != derived().cols() )
        derived().resize(rows, cols);
    ar &  (boost::serialization::make_array(derived().data(), derived().size()));
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version) {
    boost::serialization::split_member(ar, *this, file_version);
}

#endif //VALUATION_EIGENDENSEBASEADDONS_HPP
