#ifndef MAIN_HPP_
#define MAIN_HPP_

#define ELPP_NO_DEFAULT_LOG_FILE
#define ELPP_STL_LOGGING
#include "easylogging++.h"

#include <string>

//#include <RInside.h>
//#include <Rcpp.h>
//#include <RcppEigen.h>
#include <Eigen/Dense>
#include <stan/math.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "Sampler.hpp"
#include "N2_network.hpp"
#include "ER_Network.hpp"
#include "KarlFischerPaper.hpp"

#include "../test/TestMain.hpp"


INITIALIZE_EASYLOGGINGPP


#endif
