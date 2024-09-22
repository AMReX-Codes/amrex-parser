#include "amrexpr_BLassert.H"
#include "amrexpr_Gpu.H"

#ifdef AMREXPR_USE_GPU

namespace {
#ifdef AMREXPR_USE_SYCL
    sycl::device* sycl_device = nullptr;
    sycl::context* sycl_context = nullptr;
    amrexpr::gpuStream_t gpu_stream = nullptr;
#else
    amrexpr::gpuStream_t gpu_stream = 0;
#endif
}

namespace amrexpr::Gpu {

#if defined (AMREXPR_USE_SYCL)

void init_sycl (sycl::device& d, sycl::context& c, sycl::queue& q)
{
    sycl_device = &d;
    sycl_context = &c;
    gpu_stream = &q;
}

sycl::device* getSyclDevice ()
{
    AMREXPR_ALWAYS_ASSERT_WITH_MESSAGE(sycl_device,
                                     "amrexpr::Gpu::init_sycl must be called to initialize"
                                     "SYCL Device for SYCL backend");
    return sycl_device;
}

sycl::context* getSyclContext ()
{
    AMREXPR_ALWAYS_ASSERT_WITH_MESSAGE(sycl_context,
                                     "amrexpr::Gpu::init_sycl must be called to initialize"
                                     "SYCL Context for SYCL backend");
    return sycl_context;
}

#endif

void setStream (gpuStream_t a_stream)
{
    gpu_stream = a_stream;
}

[[nodiscard]] gpuStream_t getStream ()
{
#if defined (AMREXPR_USE_SYCL)
    AMREXPR_ALWAYS_ASSERT_WITH_MESSAGE(gpu_stream,
                                     "init_sycl must be called to initialize"
                                     "SYCL Queue for SYCL backend");
#endif
    return gpu_stream;
}

void streamSynchronize ()
{
#if defined(AMREXPR_USE_CUDA)
    AMREXPR_CUDA_SAFE_CALL(cudaStreamSynchronize(gpu_stream));
#elif defined(AMREXPR_USE_HIP)
    AMREXPR_HIP_SAFE_CALL(hipStreamSynchronize(gpu_stream));
#elif defined(AMREXPR_USE_SYCL)
    try {
        Gpu::getStream()->wait_and_throw();
    } catch (sycl::exception const& e) {
        throw std::runtime_error(std::string("streamSynchronize: ")+e.what());
    }
#else
    static_assert(false);
#endif
}

void htod_memcpy (void* p_d, void const* p_h, std::size_t sz)
{
#if defined(AMREXPR_USE_CUDA)
    AMREXPR_CUDA_SAFE_CALL(cudaMemcpyAsync(p_d, p_h, sz, cudaMemcpyHostToDevice,
                                         gpu_stream));
#elif defined(AMREXPR_USE_HIP)
    AMREXPR_HIP_SAFE_CALL(hipMemcpyAsync(p_d, p_h, sz, hipMemcpyHostToDevice,
                                       gpu_stream));
#elif defined(AMREXPR_USE_SYCL)
    try {
        Gpu::getStream()->submit([&] (sycl::handler& h)
        {
            h.memcpy(p_d, p_h, sz);
        });
    } catch (sycl::exception const& e) {
        throw std::runtime_error(std::string("htod_memcpy: ")+e.what());
    }
#else
    static_assert(false);
#endif
    streamSynchronize();
}

}

#endif
