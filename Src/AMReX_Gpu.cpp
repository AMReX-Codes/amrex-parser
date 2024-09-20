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

void htod_memcpy (void* p_d, void const* p_h, std::size_t sz)
{
#if defined(AMREX_USE_CUDA)
    cudaMemcpyAsync(p_d, p_h, sz, cudaMemcpyHostToDevice, gpu_stream);
    cudaStreamSynchronize(gpu_stream);
#elif defined(AMREX_USE_HIP)
#elif defined(AMREX_USE_SYCL)
#else
    static_assert(false);
#endif
}

}

#endif
