/* Copyright (C) 5/23/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#define USE_EIGEN_ACC 1
#define SINGLE_GREEKS 0
#define COARSE_CONN 5

#include <string>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;


const std::string count_str("#Samples");
const std::string rs_str("RS");
const std::string M_str("M");
const std::string assets_str("Assets");
const std::string solvent_str("Solvent");
const std::string val_str("Valuation");
const std::string delta1_str("Delta");
const std::string delta2_str("Delta using Log");
const std::string rho_str("Rho");
const std::string theta_str("Theta");
const std::string vega_str("Vega");
const std::string pi_str("Pi");
const std::string io_deg_str("In/Out degree distribution");
const std::string io_weight_str("In/Out weight distribution");
const std::string greeks_str("Greeks");

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
enum class NetworkType { ER, Fixed2D, STAR, RING, ER_SCALED, UNIFORM};

#endif
