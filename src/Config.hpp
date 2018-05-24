/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <string>

#include <boost/serialization>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

struct Config
{
public:
    fs::path output_dir;
    Config(std::string out_dir)
    {
        set_output_dir(out_dir);
    }

    /*!
     * @brief       creates new directory if not existent
     * @param path  path to directory
     * @return      1 if directory was created sucessfully, 2 if it already existed and is writable, 0 on error
     */
    int set_output_dir(std::string path)
    {
        if(boost::filesystem::exists(path))
        {
            //@TODO: check using fs::status
            return 0;
        } else {
            return boost::filesystem::create_directory(path);
        }
    }
};

#endif
