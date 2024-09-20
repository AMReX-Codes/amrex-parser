#include "AMReX_BLassert.H"
#include "AMReX_Gpu.H"

#ifdef AMREX_USE_GPU

namespace {
#ifdef AMREX_USE_SYCL
    amrex::gpuStream_t gpu_stream = nullptr;
#else
    amrex::gpuStream_t gpu_stream = 0;
#endif
}

namespace amrex::Gpu {

void setStream (gpuStream_t a_stream)
{
    gpu_stream = a_stream;
}

[[nodiscard]] gpuStream_t getStream ()
{
    return gpu_stream;
}

void streamSynchronize ()
{
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaStreamSynchronize(gpu_stream));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipStreamSynchronize(gpu_stream));
#elif defined(AMREX_USE_SYCL)
    static_assert(false);
#else
    static_assert(false);
#endif
}

void htod_memcpy (void* p_d, void const* p_h, std::size_t sz)
{
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaMemcpyAsync(p_d, p_h, sz, cudaMemcpyHostToDevice,
                                         gpu_stream));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipMemcpyAsync(p_d, p_h, sz, hipMemcpyHostToDevice,
                                       gpu_stream));
#elif defined(AMREX_USE_SYCL)
    static_assert(false);
#else
    static_assert(false);
#endif
    streamSynchronize();
}

}

#endif
