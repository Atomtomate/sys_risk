//
// Created by julian on 5/9/18.
//

#ifndef VALUATION_SAMPLER_HPP
#define VALUATION_SAMPLER_HPP

#include <iostream>
#include <functional>


#include "StatAcc.hpp"
#include "easylogging++.h"


template<class T>
class Sampler
{
private:
    std::vector<MCUtil::StatAcc<T, 100000>> accs;
    std::vector<std::function<T(void)>> observers;

    template <class Func, typename... ArgTypes>
    void call_f(Func f, ArgTypes&&... f_args)
    {
        //@TODO: use MPI producer/consumer model
        f(std::forward<ArgTypes>(f_args)...);
        for(std::size_t i=0; i < accs.size(); i++)
            accs[i](observers[i]());
    }
public:

    template <typename... ArgTypes>
    void register_observer(std::function<T(void)> f, ArgTypes&&... args)
    {
        //static_assert(std::is_same<std::invoke_result_t<Func, ArgTypes... >, T>::value, "Return type of function and accumulation type do not match!");
        observers.push_back(f);
        accs.emplace_back(std::forward<ArgTypes>(args)...);
    }

    template <class Func, class DistT, typename... ArgTypes>
    void draw_samples(Func f, DistT draw_from_dist, unsigned int n, ArgTypes&&... f_args)
    {
        for(unsigned int i = 0; i < n; i++)
        {
            //@TODO: use curry for f with f_args. see: https://stackoverflow.com/questions/152005/how-can-currying-be-done-in-c
            call_f(f, draw_from_dist(), std::forward<ArgTypes>(f_args)...);
        }
    }

    //@TODO: draw sample using some variance reduction scheme. e.g.: Controll variate, imp. sampl


    auto get_acc()
    {
        return accs;
    }

};

#endif //VALUATION_SAMPLER_HPP
