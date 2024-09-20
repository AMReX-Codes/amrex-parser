#ifndef AMREX_UTILITY_H_
#define AMREX_UTILITY_H_
#include "AMReX_Config.H"

#include <chrono>

namespace amrex
{
    using MaxResSteadyClock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                                 std::chrono::high_resolution_clock,
                                                 std::chrono::steady_clock>;

    double second ();
}

#endif
