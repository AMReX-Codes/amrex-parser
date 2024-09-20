
#ifndef AMREX_BLASSERT_H_
#define AMREX_BLASSERT_H_
#include "AMReX_Config.H"

#include "AMReX_Extension.H"
#include "AMReX_Gpu.H"
#include <cassert>
#include <stdexcept>
#include <string>

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


#if defined (AMREX_USE_CUDA)

#define AMREX_CUDA_SAFE_CALL(call) {  \
    auto amrex_i_err = call;   \
    if (cudaSuccess != amrex_i_err) { \
        std::string errStr(std::string("CUDA error in file ") + __FILE__ \
                           + " line " + std::to_string(__LINE__) \
                           + ": " + cudaGetErrorString(amrex_i_err)); \
        throw std::runtime_error(errStr); \
    }}

#elif defined (AMREX_USE_HIP)

#define AMREX_HIP_SAFE_CALL(call) { \
    auto amrex_i_err = call;         \
    if (hipSuccess != amrex_i_err) { \
        std::string errStr(std::string("HIP error in file ") + __FILE__ \
                           + " line " + std::to_string(__LINE__) \
                           + " " + hipGetErrorString(amrex_i_err)); \
        throw std::runtime_error(errStr); \
    }}

#endif

#endif
