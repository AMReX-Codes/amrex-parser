#include "AMReX_Utility.H"

namespace {
    auto clock_time_begin = amrex::MaxResSteadyClock::now();
}

double amrex::second ()
{
    return std::chrono::duration_cast<std::chrono::duration<double> >
        (amrex::MaxResSteadyClock::now() - clock_time_begin).count();
}
