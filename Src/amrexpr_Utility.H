#ifndef AMREXPR_UTILITY_H_
#define AMREXPR_UTILITY_H_
#include "amrexpr_Config.H"

#include "amrexpr_Gpu.H"
#include <algorithm>
#include <chrono>
#include <limits>

namespace amrexpr
{
    using MaxResSteadyClock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                                 std::chrono::high_resolution_clock,
                                                 std::chrono::steady_clock>;

    double second ();

    // Reference: https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
    template <typename T>
    AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    std::enable_if_t<std::is_floating_point_v<T>,bool>
    almostEqual (T x, T y, int ulp = 2)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::abs(x-y) <= std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<T>::min();
    }
}

#endif
