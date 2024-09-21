#include "amrexpr.hpp"
#include <cmath>
#include <iostream>

using namespace amrexpr;

int main (int argc, char* argv[])
{
    amrexpr::ignore_unused(argc, argv);

#if defined(AMREXPR_USE_CUDA)
    AMREXPR_CUDA_SAFE_CALL(cudaSetDevice(0));
#elif defined(AMREXPR_USE_HIP)
    AMREXPR_HIP_SAFE_CALL(hipSetDevice(0));
#elif defined(AMREXPR_USE_SYCL)
    sycl::device sycl_device(sycl::gpu_selector_v);
    sycl::context sycl_context(sycl_device);
    sycl::queue sycl_queue(sycl_context, sycl_device,
                           sycl::property_list{sycl::property::queue::in_order{}});
    amrexpr::Gpu::init_sycl(sycl_device, sycl_context, sycl_queue);
#endif

    std::size_t N = 256*256*256*8;
    auto* p = (double*)allocate_device(N*sizeof(double));

    Parser parser("epsilon/kp*2*x/w0**2*exp(-(x**2+y**2)/w0**2)*sin(k0*z)");
    parser.setConstant("epsilon",0.01);
    parser.setConstant("kp",3.5);
    parser.setConstant("w0",5.e-6);
    parser.setConstant("k0",3.e5);
    parser.registerVariables({"x","y","z"});
    auto const exe = parser.compile<3>();

    Gpu::streamSynchronize();

    double tparser;
    for (int n = 0; n < 2; ++n) { // First iteration is a warmup run.
        auto t0 = amrexpr::second();

        auto dx = 1.e-5/double(N);
        ParallelFor(N, [=] AMREXPR_GPU_DEVICE (std::size_t i)
        {
            auto x = dx*i;
            auto y = x*1.1;
            auto z = x*1.2;
            p[i] = exe(x,y,z);
        });

        Gpu::streamSynchronize();
        auto t1 = amrexpr::second();
        tparser = t1 - t0;
    }

    double tcpp;
    for (int n = 0; n < 2; ++n) { // First iteration is a warmup run.
        auto t0 = amrexpr::second();

        auto dx = 1.e-5/double(N);
        ParallelFor(N, [=] AMREXPR_GPU_DEVICE (std::size_t i)
        {
            auto x = dx*i;
            auto y = x*1.1;
            auto z = x*1.2;
            p[i] = 0.01/3.5*2*x/(5.e-6*5.e-6)*std::exp(-((x*x)+(y*y))/(5.e-6*5.e-6))*std::sin(3.e5*z);
        });

        Gpu::streamSynchronize();
        auto t1 = amrexpr::second();
        tcpp = t1 - t0;
    }

    std::cout << "Parser run time is " << std::scientific << tparser << ".\n"
              << "C++    run time is " << std::scientific << tcpp << ".\n";

    free_device(p);
}
