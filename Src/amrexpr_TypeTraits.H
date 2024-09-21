#ifndef AMREXPR_TYPETRAITS_H_
#define AMREXPR_TYPETRAITS_H_
#include "amrexpr_Config.H"

#include <type_traits>

namespace amrexpr
{
    template <typename T, typename U1, typename... Us>
    struct Same;

    template <typename T, typename U>
    struct Same<T,U>
    {
        static constexpr bool value = std::is_same_v<T,U>;
    };

    template <typename T, typename U1, typename... Us>
    struct Same
    {
        static constexpr bool value = std::is_same_v<T,U1> && Same<T,Us...>::value;
    };
}

#endif
