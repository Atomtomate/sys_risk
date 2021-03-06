/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */



#ifndef MAIN_HPP_
#define MAIN_HPP_

#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#define EIGEN_NO_STATIC_ASSERT
#endif


#define ELPP_NO_DEFAULT_LOG_FILE 1
#define ELPP_STL_LOGGING 1
#include "easylogging++.h"

#include <string>

//#include <RInside.h>
//#include <Rcpp.h>
//#include <RcppEigen.h>
//#include <stan/math.hpp>


#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>


#define EIGEN_DENSEBASE_PLUGIN "EigenDenseBaseAddons.hpp"
#include <Eigen/Dense>

#ifdef USE_MPI
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#endif

#include "Config.hpp"
#include "Sampler.hpp"
//#include "N2_network.hpp"
#include "NetwSim.hpp"
//#include "KarlFischerPaper.hpp"
#include "IOHelper.hpp"
//#include "PythonInterface.hpp"
#include "Config.hpp"


INITIALIZE_EASYLOGGINGPP


#endif
