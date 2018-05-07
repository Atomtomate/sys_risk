#ifndef STAT_ACC_HPP_
#define STAT_ACC_HPP_

#include <boost/serialization/vector.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/kurtosis.hpp>
#include <boost/accumulators/statistics/skewness.hpp>

#include <algorithm>
#include <vector>

namespace MC_util
{
template<typename T>
class StatAcc
{

using AccT = boost::accumulators::accumulator_set<T,
      boost::accumulators::features< 
        boost::accumulators::tag::mean,
        boost::accumulators::tag::variance,
        boost::accumulators::tag::skewness,
        boost::accumulators::tag::kurtosis
      > >;
public:

    enum class StatType : unsigned char {
        MEAN = 0,
        VARIANCE  = 1,
        SKEWNESS = 2,
        KURTOSIS = 3
    };

    void operator()(T val)
    {
        acc(val);
    }

    T extract(const StatType st)
    {
        T res;
        switch(st)
        {
            case StatType::MEAN:
                res = boost::accumulators::mean(acc);
                break;
            case StatType::VARIANCE:
                res = boost::accumulators::variance(acc);
                break;
            case StatType::SKEWNESS:
                res = boost::accumulators::variance(acc);
                break;
            case StatType::KURTOSIS:
                res = boost::accumulators::kurtosis(acc);
                break;
        }
        return res;
    }
private:
    AccT acc;
};
}

#endif
