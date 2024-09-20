
#ifndef AMREX_BLASSERT_H_
#define AMREX_BLASSERT_H_
#include "AMReX_Config.H"

#include "AMReX_Extension.H"
#include "AMReX_Gpu.H"
#include <cassert>

namespace amrex {

//! This shuts up the compiler about unused variables
template <class... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ignore_unused (const Ts&...) {}

void Assert_host (const char* EX, const char* file, int line, const char* msg);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Assert (const char* EX, const char* file, int line, const char* msg)
{
    AMREX_IF_ON_HOST((Assert_host(EX,file,line,msg);))
    AMREX_IF_ON_DEVICE((amrex::ignore_unused(EX,file,line,msg);))
    AMREX_IF_ON_DEVICE((assert(0);))
}

}

#define AMREX_ALWAYS_ASSERT_WITH_MESSAGE(EX,MSG) (EX)?((void)0):amrex::Assert( # EX , __FILE__, __LINE__ , # MSG)
#define AMREX_ALWAYS_ASSERT(EX) (EX)?((void)0):amrex::Assert( # EX , __FILE__, __LINE__)

#endif /*BL_BL_ASSERT_H*/
