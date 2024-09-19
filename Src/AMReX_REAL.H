#ifndef AMREX_REAL_H_
#define AMREX_REAL_H_
#include "AMReX_Config.H"

#include <cfloat>

namespace amrex {

#ifdef AMREX_USE_FLOAT
    using Real = float;
#else
    using Real = double;
#endif

}

#endif
