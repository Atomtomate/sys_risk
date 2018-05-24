/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_IOHELPER_HPP
#define VALUATION_IOHELPER_HPP

#include <boost/filesystem/fstream.hpp>
#include <boost/serialization/vector.hpp>
#include <iostream>

#include "Config.hpp"
#include "StatAcc.hpp"

namespace MCUtil {

    template<typename T>
    void write_to_csv(Sampler<T> S, Config c, std::string filename) {
        fs::path out(c.output_dir);
        out /= fs::path(filename);
        fs::path out_description(c.output_dir);
        out_description /= fs::path(filename + std::string("_description"));

        fs::ofstream ofs{out};
        fs::ofstream ofs_desc{out_description};
        auto res = S.extract(StatType::MEAN);
        for(auto el: res)
        {
            ofs_desc << el.first;
            ofs <<
        }
    }

}

#endif //VALUATION_IOHELPER_HPP
