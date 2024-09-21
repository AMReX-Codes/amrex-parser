#ifndef AMREXPR_ARENA_H_
#define AMREXPR_ARENA_H_
#include "amrexpr_Config.H"

#include <cstdlib>

namespace amrexpr
{
    [[nodiscard]] void* allocate_host (std::size_t sz);

    void free_host (void* pt);

    [[nodiscard]] void* allocate_device (std::size_t sz);

    void free_device (void* pt);
}

#endif
