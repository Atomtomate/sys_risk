/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_IOHELPER_HPP
#define VALUATION_IOHELPER_HPP


#include "Config.hpp"
#include "StatAcc.hpp"

namespace MCUtil {

    template<typename T>
    void write_to_csv(Sampler<T> S, Config c, std::string filename) {
        fs::path out(filename);
        fs::path out_description(filename + std::string("_description"));
        auto res = S.extract(StatType::MEAN);
        for(auto el: res)
        {

        }
    }

}

#endif //VALUATION_IOHELPER_HPP
