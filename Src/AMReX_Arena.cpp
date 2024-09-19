#include "AMReX_Arena.H"

namespace amrex
{

void* allocate_host (std::size_t sz)
{
    return std::malloc(sz);
}

void free_host (void* pt)
{
    std::free(pt);
}

}
