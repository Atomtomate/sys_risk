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
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>
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
    bool write_to_text(const T& data, const std::string& filename) {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        if (!ofs.is_open())
            return false;
        {
            boost::archive::text_oarchive oa(ofs);
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

    template <typename T>
    bool read_from_text(T& data, const std::string& filename) {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        if (!ifs.is_open())
            return false;
        {
            boost::archive::text_iarchive ia(ifs);
            ia >> data;
        }
        ifs.close();
        return true;
    }

    /*
    template <typename T>
    bool write_to_xml(const T& data, const std::string& filename, const std::string& data_name) {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        if (!ofs.is_open())
            return false;
        {
            boost::archive::xml_oarchive oa(ofs);
            oa << boost::serialization::make_nvp(data_name.c_str(),data);
        }
        ofs.close();
        return true;
    }

    template <typename T>
    bool read_from_xml(T& data, const std::string& filename, const std::string& data_name) {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        if (!ifs.is_open())
            return false;
        {
            boost::archive::xml_iarchive ia(ifs);
            ia >> boost::serialization::make_nvp(data_name.c_str(),data);
        }
        ifs.close();
        return true;
    }
    */


}

#endif //VALUATION_IOHELPER_HPP
