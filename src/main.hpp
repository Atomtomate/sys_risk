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

#include "Sampler.hpp"
#include "Examples.hpp"
#include "KarlFischerPaper.hpp"

#include "../test/TestMain.hpp"


INITIALIZE_EASYLOGGINGPP


#endif
