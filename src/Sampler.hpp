/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_SAMPLER_HPP
#define VALUATION_SAMPLER_HPP

#include <cmath>
#include <iostream>
#include <functional>
#include <typeinfo>
#include <type_traits>

#include "Config.hpp"
#include "StatAcc.hpp"
#include "easylogging++.h"



namespace MCUtil {

    /*!
     * @brief       This class provides methods for sampling of arbitrary functions, provided distribution and sampling function is given.
     * @tparam T    return type of the sampling function
     */
    template<typename T>
    class Sampler {
    private:
        int count;
        friend class boost::serialization::access;
        //@TODO: this needs to be dynamically adjustable between StatAcc and StatAccEigen. possibly overload StatAcc
#if USE_EIGEN_ACC
        std::vector<MCUtil::StatAccEigen<double, 0>> accs;
#else
        std::vector<MCUtil::StatAcc<double, 0>> accs;
#endif
        std::vector<std::function<T(void)>> observers;
        std::vector<std::string> descriptions;


        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            //@TODO: iterate over all stat types!
            auto res_mean = extract_plain(StatType::MEAN);
            auto res_var = extract_plain(StatType::VARIANCE);
            ar & descriptions;
            ar & res_mean;
            ar & res_var;
        }

        /*!
         * @brief           Calling this function will produce one sample
         * @tparam Func     Function type for sampling function
         * @tparam ArgTypes Argument pack for function arguments
         * @param f         Function which generates a sample
         * @param f_args
         */
        template<class Func, typename... ArgTypes>
        void call_f(Func f, ArgTypes &&... f_args) {
            auto bsn = f(std::forward<ArgTypes>(f_args)...);
            accs[0](bsn->get_scalar_allGreeks(std::forward<ArgTypes>(f_args)...));
            /* TODO: fix this workaround
             for (std::size_t i = 10; i < accs.size(); i++)
            {
                const auto res = observers[i]();
                accs[i](res);//std::move(observers[i]()));
            }
             */
        }

    public:
        Sampler &operator=(const Sampler &) = delete;

        Sampler<T>(const Sampler<T> &) = delete;

        int get_count() {return count;}


        /*!
         * @brief
         */
        Sampler(): count{0} {};

        /*!
         * @brief               Register a function to be called after each new sample is generated. Results will be accumulated and can be accessed using extract().
         * @tparam ArgTypes     Argument types for the observer function
         * @param f             Observer function
         * @param description   Description for the observer function, this is needed to identify the extracted results
         * @param args          Arguments for the observer
         */
        template<typename Func, typename... ArgTypes>
        void register_observer(Func f, const std::string &description, ArgTypes &&... args) {
            //static_assert(std::is_same<std::invoke_result_t<Func, ArgTypes... >, T>::value, "Return type of function and accumulation type do not match!");
            observers.push_back(std::move(f));
            descriptions.emplace_back(std::move(description));
            accs.emplace_back(std::forward<ArgTypes>(args)...);
        }

        /*!
         * @brief                   Draw n samples using f() and accumulate results previously registered using register_observer(). They can later be extracted using extract().
         * @tparam Func             Type of sample function
         * @tparam DistT            Type of distribution function
         * @tparam ArgTypes         Argument types for f()
         * @param f                 Function that generates sample
         * @param draw_from_dist    Distribution function
         * @param n                 Number of samples to be drawn
         * @param f_args            Arguments for f()
         */
        template<class Func, class DistT, typename... ArgTypes>
        void draw_samples(Func&& f, DistT&& draw_from_dist, unsigned int n, ArgTypes &&... f_args) {
            count += n;
            for (unsigned int i = 0; i < n; i++) {
                //@TODO: use curry for f with f_args. see: https://stackoverflow.com/questions/152005/how-can-currying-be-done-in-c
                call_f(std::forward<Func>(f), draw_from_dist(), std::forward<ArgTypes>(f_args)...);
            }
        }

        //@TODO: draw sample using some variance reduction scheme. e.g.: Controll variate, imp. sampl


        /*!
         * @brief      Extracts all results for previously registered observers.
         * @param st   Type of statistic which should be extracted. For example MCUtil::StatType::MEAN
         * @return     Returns accumulated result of type T and statistic st
         */
        auto extract_plain(const StatType st) {
            std::vector<T> res;
            for (unsigned int i = 0; i < accs.size(); i++) {
                res.emplace_back(accs[i].extract(st));
            }
            return res;
        }

        /*!
         * @brief      Extracts all results for previously registered observers.
         * @param st   Type of statistic which should be extracted. For example MCUtil::StatType::MEAN
         * @return     Returns pairs of descriptions and accumulated result of type T with statistic st
         */
        auto extract(const StatType st) {
            std::vector<std::pair<std::string, T>> res;
            for (unsigned int i = 0; i < accs.size(); i++) {
                res.emplace_back(descriptions[i], accs[i].extract(st));
            }
            return res;
        }



    };

} // end namspace

#endif //VALUATION_SAMPLER_HPP
