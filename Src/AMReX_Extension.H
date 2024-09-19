#ifndef AMREX_EXTENSION_H_
#define AMREX_EXTENSION_H_
#include "AMReX_Config.H"

#if defined(__INTEL_COMPILER) || (defined(__INTEL_LLVM_COMPILER) && !defined(SYCL_LANGUAGE_VERSION))
#define AMREX_CXX_INTEL
#elif defined(_CRAYC) || defined(__cray__)
#define AMREX_CXX_CRAY
#elif defined(__PGI)
#define AMREX_CXX_PGI
#elif defined(__NVCOMPILER)
#define AMREX_CXX_NVHPC
#elif defined(__NEC__)
#define AMREX_CXX_NEC
#elif defined(__ibmxl__)
#define AMREX_CXX_IBM
#elif defined(__clang__)
#define AMREX_CXX_CLANG
#elif defined(__GNUC__)
#define AMREX_CXX_GCC
#endif

// force inline
#if defined(__CUDA_ARCH__) && defined(AMREX_USE_CUDA)
#define AMREX_FORCE_INLINE __forceinline__

#elif defined(__HIP_DEVICE_COMPILE__) && defined(AMREX_USE_HIP)
#define AMREX_FORCE_INLINE __forceinline__

#elif defined(AMREX_CXX_INTEL)
#define AMREX_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(AMREX_CXX_CRAY)
#define AMREX_FORCE_INLINE inline

#elif defined(AMREX_CXX_PGI)
#define AMREX_FORCE_INLINE inline

#elif defined(AMREX_CXX_NVHPC)
#define AMREX_FORCE_INLINE inline

#elif defined(AMREX_CXX_NEC)
#define AMREX_FORCE_INLINE inline

#elif defined(AMREX_CXX_IBM)
#define AMREX_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(__clang__)
#define AMREX_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(__GNUC__)
#define AMREX_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(_MSC_VER)
#define AMREX_FORCE_INLINE inline __forceinline

#else
#define AMREX_FORCE_INLINE inline

#endif /* force inline */


#ifdef AMREX_USE_FORCE_INLINE
#define AMREX_INLINE AMREX_FORCE_INLINE
#else
#define AMREX_INLINE inline
#endif

// no inline
#if defined(_MSC_VER)
#define AMREX_NO_INLINE __declspec(noinline)
#elif (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIP__) || defined(__INTEL_CLANG_COMPILER))
#define AMREX_NO_INLINE __attribute__((noinline))
#else
#define AMREX_NO_INLINE
#endif

// flatten
#if defined(_MSC_VER)
#define AMREX_FLATTEN [[msvc::flatten]]
#elif defined(__clang__) || defined(__GNUC__)
#define AMREX_FLATTEN __attribute__((flatten))
#else
#define AMREX_FLATTEN
#endif

#ifdef AMREX_USE_FLATTEN_FOR
#define AMREX_ATTRIBUTE_FLATTEN_FOR AMREX_FLATTEN
#else
#define AMREX_ATTRIBUTE_FLATTEN_FOR
#endif

#endif
