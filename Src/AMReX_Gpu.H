#ifndef AMREX_GPU_H_
#define AMREX_GPU_H_
#include "AMReX_Config.H"

#include "AMReX_Extension.H"
#include <cstdlib>
#include <type_traits>

#if defined(AMREX_USE_GPU) && !defined(AMREX_USE_SYCL)

#if defined(AMREX_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(AMREX_USE_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(AMREX_USE_CUDA) && (defined(AMREX_CXX_PGI) || defined(AMREX_CXX_NVHPC))
#include <nv/target>
#define AMREX_IF_ON_DEVICE(CODE) NV_IF_TARGET(NV_IS_DEVICE, CODE)
#define AMREX_IF_ON_HOST(CODE) NV_IF_TARGET(NV_IS_HOST, CODE)
#endif

#define AMREX_GPU_HOST __host__
#define AMREX_GPU_DEVICE __device__
#define AMREX_GPU_GLOBAL __global__
#define AMREX_GPU_HOST_DEVICE __host__ __device__
#define AMREX_GPU_CONSTANT __constant__

#define AMREX_GPU_MANAGED __managed__
#define AMREX_GPU_DEVICE_MANAGED __device__ __managed__

#else

#define AMREX_GPU_HOST
#define AMREX_GPU_DEVICE
#define AMREX_GPU_GLOBAL
#define AMREX_GPU_HOST_DEVICE
#define AMREX_GPU_CONSTANT
#define AMREX_GPU_MANAGED
#define AMREX_GPU_DEVICE_MANAGED

#endif

#define AMREX_DEVICE_COMPILE (__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__ || __SYCL_DEVICE_ONLY__)

// Remove surrounding parentheses if present
#define AMREX_IMPL_STRIP_PARENS(X) AMREX_IMPL_ESC(AMREX_IMPL_ISH X)
#define AMREX_IMPL_ISH(...) AMREX_IMPL_ISH __VA_ARGS__
#define AMREX_IMPL_ESC(...) AMREX_IMPL_ESC_(__VA_ARGS__)
#define AMREX_IMPL_ESC_(...) AMREX_IMPL_VAN_##__VA_ARGS__
#define AMREX_IMPL_VAN_AMREX_IMPL_ISH

#if !defined(AMREX_IF_ON_DEVICE) && !defined(AMREX_IF_ON_HOST)
#if (defined(AMREX_USE_CUDA) && defined(__CUDA_ARCH__)) ||         \
    (defined(AMREX_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)) || \
    (defined(AMREX_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
#define AMREX_IF_ON_DEVICE(CODE) \
  { AMREX_IMPL_STRIP_PARENS(CODE) }
#define AMREX_IF_ON_HOST(CODE) \
  {}
#else
#define AMREX_IF_ON_DEVICE(CODE) \
  {}
#define AMREX_IF_ON_HOST(CODE) \
  { AMREX_IMPL_STRIP_PARENS(CODE) }
#endif
#endif

#ifdef AMREX_USE_SYCL
# include <sycl/sycl.hpp>
#endif

#define AMREX_WRONG_NUM_ARGS(...) static_assert(false,"Wrong number of arguments to macro")

#define AMREX_GET_DGV_MACRO(_1,_2,_3,NAME,...) NAME
#define AMREX_DEVICE_GLOBAL_VARIABLE(...) AMREX_GET_DGV_MACRO(__VA_ARGS__,\
                                              AMREX_DGVARR, AMREX_DGV,\
                                              AMREX_WRONG_NUM_ARGS)(__VA_ARGS__)

namespace amrex {

#ifdef AMREX_USE_GPU

#if defined(AMREX_USE_CUDA)
using gpuStream_t = cudaStream_t;
#elif defined(AMREX_USE_HIP)
using gpuStream_t = hipStream_t;
#elif defined(AMREX_USE_SYCL)
using gpuStream_t = sycl::queue*;
#endif

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

template<int amrex_launch_bounds_max_threads, class L>
__launch_bounds__(amrex_launch_bounds_max_threads)
__global__ void launch_global (L f0) { f0(); }

#endif

namespace Gpu {

    void setStream (gpuStream_t a_stream);

    gpuStream_t getStream ();

    void htod_memcpy (void* p_d, void const* p_h, std::size_t sz);

    void streamSynchronize ();

#if defined (AMREX_USE_SYCL)
    void init_sycl (sycl::device& d, sycl::context& c, sycl::queue& q);
    sycl::device* getSyclDevice ();
    sycl::context* getSyclContext ();
#endif
}

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

template <typename T, typename L, typename M=std::enable_if_t<std::is_integral_v<T>> >
void ParallelFor (T n, L const& f)
{
    if (n <= 0) { return; }

    constexpr unsigned int max_threads = 256;
    auto nblocks = static_cast<unsigned int>((n+(max_threads-1)) / max_threads);
    launch_global<256><<<nblocks,max_threads,0,Gpu::getStream()>>>(
        [=] AMREX_GPU_DEVICE
        {
            auto i = T(blockDim.x*blockIdx.x+threadIdx.x);
            if (i < n) { f(i); }
        });
}

#elif defined(AMREX_USE_SYCL)

template <typename T, typename L, typename M=std::enable_if_t<std::is_integral_v<T>> >
void ParallelFor (T n, L const& f)
{
    if (n <= 0) { return; }

    auto& q = *Gpu::getStream();
    try {
        q.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n),
                           [=] (sycl::item<1> item)
            {
                f(T(item.get_linear_id()));
            });
        });
    } catch (sycl::exception const& e) {
        throw std::runtime_error(std::string("ParallelFor: ")+e.what());
    }
}

#endif

#else /* CPU */

namespace Gpu {
    inline void streamSynchronize () {}
}

template <typename T, typename L, typename M=std::enable_if_t<std::is_integral_v<T>> >
AMREX_ATTRIBUTE_FLATTEN_FOR
void ParallelFor (T n, L const& f)
{
    for (T i = 0; i < n; ++i) { f(i); }
}

#endif

}

#endif
