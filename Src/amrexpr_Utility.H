#ifndef AMREXPR_UTILITY_H_
#define AMREXPR_UTILITY_H_
#include "amrexpr_Config.H"

#include <chrono>

namespace amrexpr
{
    using MaxResSteadyClock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                                 std::chrono::high_resolution_clock,
                                                 std::chrono::steady_clock>;

    double second ();
}

#endif
