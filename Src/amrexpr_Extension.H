#ifndef AMREXPR_EXTENSION_H_
#define AMREXPR_EXTENSION_H_
#include "amrexpr_Config.H"

#if defined(__INTEL_COMPILER) || (defined(__INTEL_LLVM_COMPILER) && !defined(SYCL_LANGUAGE_VERSION))
#define AMREXPR_CXX_INTEL
#elif defined(_CRAYC) || defined(__cray__)
#define AMREXPR_CXX_CRAY
#elif defined(__PGI)
#define AMREXPR_CXX_PGI
#elif defined(__NVCOMPILER)
#define AMREXPR_CXX_NVHPC
#elif defined(__NEC__)
#define AMREXPR_CXX_NEC
#elif defined(__ibmxl__)
#define AMREXPR_CXX_IBM
#elif defined(__clang__)
#define AMREXPR_CXX_CLANG
#elif defined(__GNUC__)
#define AMREXPR_CXX_GCC
#endif

// force inline
#if defined(__CUDA_ARCH__) && defined(AMREXPR_USE_CUDA)
#define AMREXPR_FORCE_INLINE __forceinline__

#elif defined(__HIP_DEVICE_COMPILE__) && defined(AMREXPR_USE_HIP)
#define AMREXPR_FORCE_INLINE __forceinline__

#elif defined(AMREXPR_CXX_INTEL)
#define AMREXPR_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(AMREXPR_CXX_CRAY)
#define AMREXPR_FORCE_INLINE inline

#elif defined(AMREXPR_CXX_PGI)
#define AMREXPR_FORCE_INLINE inline

#elif defined(AMREXPR_CXX_NVHPC)
#define AMREXPR_FORCE_INLINE inline

#elif defined(AMREXPR_CXX_NEC)
#define AMREXPR_FORCE_INLINE inline

#elif defined(AMREXPR_CXX_IBM)
#define AMREXPR_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(__clang__)
#define AMREXPR_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(__GNUC__)
#define AMREXPR_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(_MSC_VER)
#define AMREXPR_FORCE_INLINE inline __forceinline

#else
#define AMREXPR_FORCE_INLINE inline

#endif /* force inline */

// no inline
#if defined(_MSC_VER)
#define AMREXPR_NO_INLINE __declspec(noinline)
#elif (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIP__) || defined(__INTEL_CLANG_COMPILER))
#define AMREXPR_NO_INLINE __attribute__((noinline))
#else
#define AMREXPR_NO_INLINE
#endif

// flatten
#if defined(_MSC_VER)
#define AMREXPR_FLATTEN [[msvc::flatten]]
#elif defined(__clang__) || defined(__GNUC__)
#define AMREXPR_FLATTEN __attribute__((flatten))
#else
#define AMREXPR_FLATTEN
#endif

#define AMREXPR_ATTRIBUTE_FLATTEN_FOR AMREXPR_FLATTEN

#endif
