#ifndef AMREX_ARENA_H_
#define AMREX_ARENA_H_
#include "AMReX_Config.H"

#include <cstdlib>

namespace amrex
{
    [[nodiscard]] void* allocate_host (std::size_t sz);

    void free_host (void* pt);

#ifdef AMREX_USE_GPU
    [[nodiscard]] void* allocate_device (std::size_t sz);

    void free_device (void* pt);
#endif
}

#endif
