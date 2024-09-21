#ifndef AMREXPR_GPU_H_
#define AMREXPR_GPU_H_
#include "amrexpr_Config.H"

#include "amrexpr_Extension.H"
#include <cstdlib>
#include <type_traits>

#if defined(AMREXPR_USE_GPU) && !defined(AMREXPR_USE_SYCL)

#if defined(AMREXPR_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(AMREXPR_USE_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(AMREXPR_USE_CUDA) && (defined(AMREXPR_CXX_PGI) || defined(AMREXPR_CXX_NVHPC))
#include <nv/target>
#define AMREXPR_IF_ON_DEVICE(CODE) NV_IF_TARGET(NV_IS_DEVICE, CODE)
#define AMREXPR_IF_ON_HOST(CODE) NV_IF_TARGET(NV_IS_HOST, CODE)
#endif

#define AMREXPR_GPU_HOST __host__
#define AMREXPR_GPU_DEVICE __device__
#define AMREXPR_GPU_GLOBAL __global__
#define AMREXPR_GPU_HOST_DEVICE __host__ __device__
#define AMREXPR_GPU_CONSTANT __constant__

#define AMREXPR_GPU_MANAGED __managed__
#define AMREXPR_GPU_DEVICE_MANAGED __device__ __managed__

#else

#define AMREXPR_GPU_HOST
#define AMREXPR_GPU_DEVICE
#define AMREXPR_GPU_GLOBAL
#define AMREXPR_GPU_HOST_DEVICE
#define AMREXPR_GPU_CONSTANT
#define AMREXPR_GPU_MANAGED
#define AMREXPR_GPU_DEVICE_MANAGED

#endif

#define AMREXPR_DEVICE_COMPILE (__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__ || __SYCL_DEVICE_ONLY__)

// Remove surrounding parentheses if present
#define AMREXPR_IMPL_STRIP_PARENS(X) AMREXPR_IMPL_ESC(AMREXPR_IMPL_ISH X)
#define AMREXPR_IMPL_ISH(...) AMREXPR_IMPL_ISH __VA_ARGS__
#define AMREXPR_IMPL_ESC(...) AMREXPR_IMPL_ESC_(__VA_ARGS__)
#define AMREXPR_IMPL_ESC_(...) AMREXPR_IMPL_VAN_##__VA_ARGS__
#define AMREXPR_IMPL_VAN_AMREXPR_IMPL_ISH

#if !defined(AMREXPR_IF_ON_DEVICE) && !defined(AMREXPR_IF_ON_HOST)
#if (defined(AMREXPR_USE_CUDA) && defined(__CUDA_ARCH__)) ||         \
    (defined(AMREXPR_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)) || \
    (defined(AMREXPR_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
#define AMREXPR_IF_ON_DEVICE(CODE) \
  { AMREXPR_IMPL_STRIP_PARENS(CODE) }
#define AMREXPR_IF_ON_HOST(CODE) \
  {}
#else
#define AMREXPR_IF_ON_DEVICE(CODE) \
  {}
#define AMREXPR_IF_ON_HOST(CODE) \
  { AMREXPR_IMPL_STRIP_PARENS(CODE) }
#endif
#endif

#ifdef AMREXPR_USE_SYCL
# include <sycl/sycl.hpp>
#endif

#define AMREXPR_WRONG_NUM_ARGS(...) static_assert(false,"Wrong number of arguments to macro")

#define AMREXPR_GET_DGV_MACRO(_1,_2,_3,NAME,...) NAME
#define AMREXPR_DEVICE_GLOBAL_VARIABLE(...) AMREXPR_GET_DGV_MACRO(__VA_ARGS__,\
                                              AMREXPR_DGVARR, AMREXPR_DGV,\
                                              AMREXPR_WRONG_NUM_ARGS)(__VA_ARGS__)

namespace amrexpr {

#ifdef AMREXPR_USE_GPU

#if defined(AMREXPR_USE_CUDA)
using gpuStream_t = cudaStream_t;
#elif defined(AMREXPR_USE_HIP)
using gpuStream_t = hipStream_t;
#elif defined(AMREXPR_USE_SYCL)
using gpuStream_t = sycl::queue*;
#endif

#if defined(AMREXPR_USE_CUDA) || defined(AMREXPR_USE_HIP)

template<int amrexpr_launch_bounds_max_threads, class L>
__launch_bounds__(amrexpr_launch_bounds_max_threads)
__global__ void launch_global (L f0) { f0(); }

#endif

namespace Gpu {

    void setStream (gpuStream_t a_stream);

    gpuStream_t getStream ();

    void htod_memcpy (void* p_d, void const* p_h, std::size_t sz);

    void streamSynchronize ();

#if defined (AMREXPR_USE_SYCL)
    void init_sycl (sycl::device& d, sycl::context& c, sycl::queue& q);
    sycl::device* getSyclDevice ();
    sycl::context* getSyclContext ();
#endif
}

#if defined(AMREXPR_USE_CUDA) || defined(AMREXPR_USE_HIP)

template <typename T, typename L, typename M=std::enable_if_t<std::is_integral_v<T>> >
void ParallelFor (T n, L const& f)
{
    if (n <= 0) { return; }

    constexpr unsigned int max_threads = 256;
    auto nblocks = static_cast<unsigned int>((n+(max_threads-1)) / max_threads);
    launch_global<256><<<nblocks,max_threads,0,Gpu::getStream()>>>(
        [=] AMREXPR_GPU_DEVICE
        {
            auto i = T(blockDim.x*blockIdx.x+threadIdx.x);
            if (i < n) { f(i); }
        });
}

#elif defined(AMREXPR_USE_SYCL)

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
AMREXPR_ATTRIBUTE_FLATTEN_FOR
void ParallelFor (T n, L const& f)
{
    for (T i = 0; i < n; ++i) { f(i); }
}

#endif

}

#endif
