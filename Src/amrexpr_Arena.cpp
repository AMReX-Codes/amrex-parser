#include "amrexpr_Arena.H"
#include "amrexpr_BLassert.H"
#include "amrexpr_Gpu.H"

namespace amrexpr
{

void* allocate_host (std::size_t sz)
{
#if defined(AMREXPR_USE_CUDA)
    void* p;
    AMREXPR_CUDA_SAFE_CALL(cudaHostAlloc(&p, sz, cudaHostAllocMapped));
    return p;
#elif defined(AMREXPR_USE_HIP)
    void* p;
    AMREXPR_HIP_SAFE_CALL(hipHostMalloc(&p, sz, hipHostMallocMapped |
                                      hipHostMallocNonCoherent));
    return p;
#elif defined(AMREXPR_USE_SYCL)
    return sycl::malloc_host(sz, *Gpu::getSyclContext());
#else
    return std::malloc(sz);
#endif
}

void free_host (void* pt)
{
#if defined(AMREXPR_USE_CUDA)
    AMREXPR_CUDA_SAFE_CALL(cudaFreeHost(pt));
#elif defined(AMREXPR_USE_HIP)
    AMREXPR_HIP_SAFE_CALL(hipHostFree(pt));
#elif defined(AMREXPR_USE_SYCL)
    sycl::free(pt, *Gpu::getSyclContext());
#else
    std::free(pt);
#endif
}

void* allocate_device (std::size_t sz)
{
    void* p;
#if defined(AMREXPR_USE_CUDA)
    AMREXPR_CUDA_SAFE_CALL(cudaMalloc(&p, sz));
#elif defined(AMREXPR_USE_HIP)
    AMREXPR_HIP_SAFE_CALL(hipMalloc(&p, sz));
#elif defined(AMREXPR_USE_SYCL)
    p = sycl::malloc_device(sz, *Gpu::getSyclDevice(), *Gpu::getSyclContext());
#else
    p = std::malloc(sz);
#endif
    return p;
}

void free_device (void* pt)
{
#if defined(AMREXPR_USE_CUDA)
    AMREXPR_CUDA_SAFE_CALL(cudaFree(pt));
#elif defined(AMREXPR_USE_HIP)
    AMREXPR_HIP_SAFE_CALL(hipFree(pt));
#elif defined(AMREXPR_USE_SYCL)
    sycl::free(pt, *Gpu::getSyclContext());
#else
    std::free(pt);
#endif
}

}
