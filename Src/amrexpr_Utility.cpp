#include "amrexpr_Utility.H"

namespace {
    auto clock_time_begin = amrexpr::MaxResSteadyClock::now();
}

double amrexpr::second ()
{
    return std::chrono::duration_cast<std::chrono::duration<double> >
        (amrexpr::MaxResSteadyClock::now() - clock_time_begin).count();
}
