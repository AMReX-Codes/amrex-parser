#include "AMReX_Arena.H"
#include "AMReX_Gpu.H"

namespace amrex
{

void* allocate_host (std::size_t sz)
{
#if defined(AMREX_USE_CUDA)
    void* p;
    cudaHostAlloc(&p, sz, cudaHostAllocMapped);
    return p;
#elif defined(AMREX_USE_HIP)
    void* p;
    hipHostAlloc(&p, sz, hipHostAllocMapped | hipHostMallocNonCoherent);
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
    cudaFreeHost(pt);
#elif defined(AMREX_USE_HIP)
    hipHostFree(pt);
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
    cudaMalloc(&p, sz);
#elif defined(AMREX_USE_HIP)
    hipMalloc(&p, sz);
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
    cudaFree(pt);
#elif defined(AMREX_USE_HIP)
    hipFree(pt);
#elif defined(AMREX_USE_SYCL)
    sycl::free(...);
#else
    std::free(pt);
#endif
}

}
