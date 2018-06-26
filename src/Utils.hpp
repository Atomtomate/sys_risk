/* Copyright (C) 6/22/18 Julian Stobbe - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file.
 */

#ifndef VALUATION_UTILS_HPP
#define VALUATION_UTILS_HPP

namespace Utils
{

    // https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
    template <class T>
    constexpr
    std::string_view
    type_name()
    {
        using namespace std;
#ifdef __clang__
        string_view p = __PRETTY_FUNCTION__;
        return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
        string_view p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
        return string_view(p.data() + 36, p.size() - 36 - 1);
#  else
        return string_view(p.data() + 49, p.find(';', 49) - 49);
#  endif
#elif defined(_MSC_VER)
        string_view p = __FUNCSIG__;
    return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
    }
}


#endif //VALUATION_UTILS_HPP
