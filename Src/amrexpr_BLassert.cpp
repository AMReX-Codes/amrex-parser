#include "amrexpr_BLassert.H"

#include <cstdio>

namespace amrexpr
{

void Assert_host (const char* EX, const char* file, int line, const char* msg)
{
    const int N = 512;
    char buf[N];
    std::snprintf(buf, N, "Assertion `%s' failed, file \"%s\", line %d, Msg: %s",
                  EX, file, line, msg);
    throw std::runtime_error(buf);
}

}
