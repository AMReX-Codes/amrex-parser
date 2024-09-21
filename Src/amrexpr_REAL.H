#ifndef AMREXPR_REAL_H_
#define AMREXPR_REAL_H_
#include "amrexpr_Config.H"

#include <cfloat>

namespace amrexpr {

#ifdef AMREXPR_USE_FLOAT
    using Real = float;
#else
    using Real = double;
#endif

}

#endif
