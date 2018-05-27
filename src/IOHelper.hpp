/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * partly taken from: https://stackoverflow.com/questions/18382457/eigen-and-boostserialize
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_IOHELPER_HPP
#define VALUATION_IOHELPER_HPP

#include <boost/filesystem/fstream.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <iostream>

#include "Config.hpp"
#include "StatAcc.hpp"
#include "EigenDenseBaseAddons.hpp"

namespace MCUtil {

    template <typename T>
    bool write_to_binary(const T& data, const std::string& filename) {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        if (!ofs.is_open())
            return false;
        {
            boost::archive::binary_oarchive oa(ofs);
            oa << data;
        }
        ofs.close();
        return true;
    }

    template <typename T>
    bool read_from_binary(T& data, const std::string& filename) {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        if (!ifs.is_open())
            return false;
        {
            boost::archive::binary_iarchive ia(ifs);
            ia >> data;
        }
        ifs.close();
        return true;
    }


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
            ofs << S;
        }
    }

}

#endif //VALUATION_IOHELPER_HPP
