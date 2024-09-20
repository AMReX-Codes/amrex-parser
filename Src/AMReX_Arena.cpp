#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Gpu.H"

namespace amrex
{

void* allocate_host (std::size_t sz)
{
#if defined(AMREX_USE_CUDA)
    void* p;
    AMREX_CUDA_SAFE_CALL(cudaHostAlloc(&p, sz, cudaHostAllocMapped));
    return p;
#elif defined(AMREX_USE_HIP)
    void* p;
    AMREX_HIP_SAFE_CALL(hipHostMalloc(&p, sz, hipHostMallocMapped |
                                      hipHostMallocNonCoherent));
    return p;
#elif defined(AMREX_USE_SYCL)
    return sycl::malloc_host(...);
#else
    return std::malloc(sz);
#endif
}

void free_host (void* pt)
{
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaFreeHost(pt));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipHostFree(pt));
#elif defined(AMREX_USE_SYCL)
    sycl::free(...);
#else
    std::free(pt);
#endif
}

void* allocate_device (std::size_t sz)
{
    void* p;
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaMalloc(&p, sz));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipMalloc(&p, sz));
#elif defined(AMREX_USE_SYCL)
    p = sycl::malloc_device(...);
#else
    p = std::malloc(sz);
#endif
    return p;
}

void free_device (void* pt)
{
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaFree(pt));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipFree(pt));
#elif defined(AMREX_USE_SYCL)
    sycl::free(...);
#else
    std::free(pt);
#endif
}

}
