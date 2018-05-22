#ifndef STAT_ACC_HPP_
#define STAT_ACC_HPP_

#include <deque>

#include <boost/accumulators/numeric/functional/vector.hpp>
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

namespace MCUtil
{


    /*!
     * @brief Statistic types to be collected
     */
enum class StatType : unsigned char {
    MEAN = 0,
    VARIANCE  = 1,
    SKEWNESS = 2,
    KURTOSIS = 3
};

//use cache with sfinae to only define caching vector if needed

    /*!
     * @brief               This class provides the low level accumulation functionality for statistical quantities of observables
     * @tparam T            Type of quantity
     * @tparam CACHE_SIZE   Cache for the last CACHE_SIZE samples. This can be used for quantities that depend on multiple samples
     */
template<typename T, unsigned long CACHE_SIZE = 0>
class StatAcc
{

using AccT = boost::accumulators::accumulator_set<T,
      boost::accumulators::features< 
        boost::accumulators::tag::mean,
        boost::accumulators::tag::variance
        //,boost::accumulators::tag::skewness
        //,boost::accumulators::tag::kurtosis
      > >;

private:
    AccT acc;
    //@TODO: switch to boost circular buffer to store full time series?
    std::deque<T> cache;
    unsigned long cache_used;

public:
    /*StatAcc(): cache_used{0}
    {
        if(CACHE_SIZE)
            cache.resize(CACHE_SIZE);
    }
     */

    StatAcc &operator=(const StatAcc &) = delete;
    //StatAcc(const StatAcc&) = delete;

    /*!
     * @brief
     * @tparam ArgTypes     Argument types for the constructor of the quantity type to be accumulated
     * @param args          Arguments for the constructor of the quantity type to be accumulated
     */
    template<typename... ArgTypes>
    StatAcc(ArgTypes&&... args):
        acc(T(std::forward<ArgTypes>(args)...)), cache_used{0}
    {
        if(CACHE_SIZE)
            cache.resize(CACHE_SIZE);
    }

    friend std::ostream &operator<<(std::ostream &stream, const StatAcc &sa) {
        stream << "This is the temporary output for StatAcc";
        return stream;
    }

    /*!
     * @brief       Adds another sample to the accumulator
     * @param val   Value of next sample
     */
    void operator()(T val)
    {
        if(CACHE_SIZE)
        {
            if(cache_used < CACHE_SIZE)
            {
                cache.push_back(val);
                cache_used += 1;
            }
            else
            {
                cache.pop_front();
                cache.push_back(val);
            }
        }
        acc(val);
    }

    /*!
     * @brief       extracts statistic of accumulated quantity
     * @param st    Stat Type
     * @return      Value of statistic for quantity
     */
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
                /* @TODO: template disable
                 * case StatType::SKEWNESS:
                    res = boost::accumulators::variance(acc);
                    break;
                case StatType::KURTOSIS:
                    res = boost::accumulators::kurtosis(acc);
                    break;
                    */
        }
        return res;
    }
};
}

#endif