
#ifndef AMREXPR_BLASSERT_H_
#define AMREXPR_BLASSERT_H_
#include "amrexpr_Config.H"

#include "amrexpr_Extension.H"
#include "amrexpr_Gpu.H"
#include <cassert>
#include <stdexcept>
#include <string>

namespace amrexpr {

//! This shuts up the compiler about unused variables
template <class... Ts>
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
void ignore_unused (const Ts&...) {}

void Assert_host (const char* EX, const char* file, int line, const char* msg);

AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
void Assert (const char* EX, const char* file, int line, const char* msg)
{
    AMREXPR_IF_ON_HOST((Assert_host(EX,file,line,msg);))
    AMREXPR_IF_ON_DEVICE((amrexpr::ignore_unused(EX,file,line,msg);))
    AMREXPR_IF_ON_DEVICE((assert(0);))
}

}

#define AMREXPR_ALWAYS_ASSERT_WITH_MESSAGE(EX,MSG) (EX)?((void)0):amrexpr::Assert( # EX , __FILE__, __LINE__ , # MSG)
#define AMREXPR_ALWAYS_ASSERT(EX) (EX)?((void)0):amrexpr::Assert( # EX , __FILE__, __LINE__)


#if defined (AMREXPR_USE_CUDA)

#define AMREXPR_CUDA_SAFE_CALL(call) {  \
    auto amrexpr_i_err = call;   \
    if (cudaSuccess != amrexpr_i_err) { \
        std::string errStr(std::string("CUDA error in file ") + __FILE__ \
                           + " line " + std::to_string(__LINE__) \
                           + ": " + cudaGetErrorString(amrexpr_i_err)); \
        throw std::runtime_error(errStr); \
    }}

#elif defined (AMREXPR_USE_HIP)

#define AMREXPR_HIP_SAFE_CALL(call) { \
    auto amrexpr_i_err = call;         \
    if (hipSuccess != amrexpr_i_err) { \
        std::string errStr(std::string("HIP error in file ") + __FILE__ \
                           + " line " + std::to_string(__LINE__) \
                           + " " + hipGetErrorString(amrexpr_i_err)); \
        throw std::runtime_error(errStr); \
    }}

#endif

#endif
