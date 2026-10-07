#ifndef AMREXPR_MATH_H_
#define AMREXPR_MATH_H_
#include "amrexpr_Config.H"

#include "amrexpr_Extension.H"
#include "amrexpr_Gpu.H"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#ifdef AMREXPR_USE_SYCL
#  include <sycl/sycl.hpp>
#endif

namespace amrexpr { inline namespace disabled {
    // If it is inside namespace amrexpr, or amrexpr namespace is imported with using namespace amrexpr or
    // amrexpr::disabled, unqualified abs functions are disabled with a compile time error such as,
    // call of overload abs(int&) is ambiguous, or a link time error such as, undefined reference to
    // `amrexpr::disabled::abs(double)'.  To fix it, one can use `std::abs` or `amrexpr::Math::abs`.
    // The latter works in both host and device functions, whereas `std::abs` does not currently
    // work on device with HIP and SYCL.
    AMREXPR_GPU_HOST_DEVICE double abs (double);
    AMREXPR_GPU_HOST_DEVICE float abs (float);
    AMREXPR_GPU_HOST_DEVICE long double abs (long double);
    AMREXPR_GPU_HOST_DEVICE int abs (int);
    AMREXPR_GPU_HOST_DEVICE long abs (long);
    AMREXPR_GPU_HOST_DEVICE long long abs (long long);
}}

namespace amrexpr::Math {

template <typename T>
constexpr std::enable_if_t<std::is_floating_point_v<T>,T> pi ()
{
    return T(3.1415926535897932384626433832795029L);
}

//! Return cos(x*pi) given x
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
double cospi (double x)
{
#if defined(AMREXPR_USE_SYCL)
    return sycl::cospi(x);
#else
    AMREXPR_IF_ON_DEVICE(( return ::cospi(x); ))
    AMREXPR_IF_ON_HOST(( return std::cos(pi<double>()*x); ))
#endif
}

//! Return cos(x*pi) given x
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
float cospi (float x)
{
#if defined(AMREXPR_USE_SYCL)
    return sycl::cospi(x);
#else
    AMREXPR_IF_ON_DEVICE(( return ::cospif(x); ))
    AMREXPR_IF_ON_HOST(( return std::cos(pi<float>()*x); ))
#endif
}

//! Return sin(x*pi) given x
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
double sinpi (double x)
{
#if defined(AMREXPR_USE_SYCL)
    return sycl::sinpi(x);
#else
    AMREXPR_IF_ON_DEVICE(( return ::sinpi(x); ))
    AMREXPR_IF_ON_HOST(( return std::sin(pi<double>()*x); ))
#endif
}

//! Return sin(x*pi) given x
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
float sinpi (float x)
{
#if defined(AMREXPR_USE_SYCL)
    return sycl::sinpi(x);
#else
    AMREXPR_IF_ON_DEVICE(( return ::sinpif(x); ))
    AMREXPR_IF_ON_HOST(( return std::sin(pi<float>()*x); ))
#endif
}

namespace detail {
    AMREXPR_FORCE_INLINE void sincos (double x, double* sinx, double* cosx) {
#if defined(_GNU_SOURCE) && !defined(__APPLE__)
        ::sincos(x, sinx, cosx);
#else
        *sinx = std::sin(x);
        *cosx = std::cos(x);
#endif
    }

    AMREXPR_FORCE_INLINE void sincosf (float x, float* sinx, float* cosx) {
#if defined(_GNU_SOURCE) && !defined(__APPLE__)
        ::sincosf(x, sinx, cosx);
#else
        *sinx = std::sin(x);
        *cosx = std::cos(x);
#endif
    }
}

//! Return sine and cosine of given number
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
std::pair<double,double> sincos (double x)
{
    std::pair<double,double> r;
#if defined(AMREXPR_USE_SYCL)
    r.first = sycl::sincos(x, sycl::private_ptr<double>(&r.second));
#else
    AMREXPR_IF_ON_DEVICE(( ::sincos(x, &r.first, &r.second); ))
    AMREXPR_IF_ON_HOST(( detail::sincos(x, &r.first, &r.second); ))
#endif
    return r;
}

//! Return sine and cosine of given number
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
std::pair<float,float> sincos (float x)
{
    std::pair<float,float> r;
#if defined(AMREXPR_USE_SYCL)
    r.first = sycl::sincos(x, sycl::private_ptr<float>(&r.second));
#else
    AMREXPR_IF_ON_DEVICE(( ::sincosf(x, &r.first, &r.second); ))
    AMREXPR_IF_ON_HOST(( detail::sincosf(x, &r.first, &r.second); ))
#endif
    return r;
}

//! Return sin(pi*x) and cos(pi*x) given x
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
std::pair<double,double> sincospi (double x)
{
    std::pair<double,double> r;
#if defined(AMREXPR_USE_SYCL)
    r = sincos(pi<double>()*x);
#else
    AMREXPR_IF_ON_DEVICE(( ::sincospi(x, &r.first, &r.second); ))
    AMREXPR_IF_ON_HOST(( r = sincos(pi<double>()*x); ))
#endif
    return r;
}

//! Return sin(pi*x) and cos(pi*x) given x
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
std::pair<float,float> sincospi (float x)
{
    std::pair<float,float> r;
#if defined(AMREXPR_USE_SYCL)
    r = sincos(pi<float>()*x);
#else
    AMREXPR_IF_ON_DEVICE(( ::sincospif(x, &r.first, &r.second); ))
    AMREXPR_IF_ON_HOST(( r = sincos(pi<float>()*x); ))
#endif
    return r;
}

//! Return pow(x, Power), where Power is an integer known at compile time
template <int Power, typename T,
    typename = std::enable_if_t<!std::is_integral<T>() || Power>=0>>
AMREXPR_FORCE_INLINE
constexpr T powi (T x) noexcept
{
    if constexpr (Power < 0) {
        return T(1)/powi<-Power>(x);
    } else if constexpr (Power == 0) {
        //note: 0^0 is implementation-defined, but most compilers return 1
        return T(1);
    } else if constexpr (Power == 1) {
        return x;
    } else if constexpr (Power == 2) {
        return x*x;
    } else if constexpr (Power%2 == 0) {
        return powi<2>(powi<Power/2>(x));
    } else {
        return x*powi<Power-1>(x);
    }
}

template <typename T>
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
T comp_ellint_1 (T k)
{
    // Computing K based on DLMF
    // https://dlmf.nist.gov/19.8
    T tol = std::numeric_limits<T>::epsilon();

    T a0 = 1.0;
    T g0 = std::sqrt(1.0 - k*k);
    T a = a0;
    T g = g0;

    // Find Arithmetic Geometric mean
    while(std::abs(a0 - g0) > tol) {
        a = 0.5*(a0 + g0);
        g = std::sqrt(a0 * g0);

        a0 = a;
        g0 = g;
    }

    return 0.5*pi<T>()/a;
}

template <typename T>
AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
T comp_ellint_2 (T k)
{
    // Computing E based on DLMF
    // https://dlmf.nist.gov/19.8
    T Kcomp = amrexpr::Math::comp_ellint_1<T>(k);
    T tol = std::numeric_limits<T>::epsilon();

    // Step Zero
    T a0 = 1.0;
    T g0 = std::sqrt(1.0 - k*k);
    T cn = std::sqrt(a0*a0 - g0*g0);

    // Step 1
    int n = 1;
    T a = 0.5 * (a0 + g0);
    T g = std::sqrt(a0*g0);
    cn = 0.25*cn*cn/a;

    T sum_val = a*a;
    a0 = a;
    g0 = g;

    while(std::abs(cn*cn) > tol) {
        // Compute coefficients for this iteration
        a = 0.5 * (a0 + g0);
        g = std::sqrt(a0*g0);
        cn = 0.25*cn*cn/a;

        n++;
        sum_val -= std::pow(2,n-1)*cn*cn;

        // Save a and g for next iteration
        a0 = a;
        g0 = g;
    }

    return Kcomp*sum_val;
}

// Under fast math (e.g., -ffast-math, -ffinite-math-only, /fp:fast, and Intel's
// SYCL compiler by default), std::isnan etc. may always evaluate to false.  The
// functions below work in all modes.  In fast-math modes, they test the bit
// pattern, hidden from the optimizer so that it cannot fold the test away.

// Clang: no macro tells if -fno-honor-nans or -fno-honor-infinities is on.
// _WIN32: std::isnan is host only for nvcc with MSVC.
#if (defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ == 1)) \
    || defined(__FAST_MATH__) || defined(_M_FP_FAST) || defined(_WIN32) \
    || defined(__clang__)
#  define AMREXPR_MATH_ISNAN_BITS 1
#endif

namespace detail {
    //! Return v, opaque to the optimizer under fast math.  Otherwise, clang
    //! assumes that the result of a fast-math operation is not NaN or inf, and
    //! folds a bit test of it to a constant.
    template <typename U>
    AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    U fp_opaque (U v) noexcept
    {
#if defined(AMREXPR_MATH_ISNAN_BITS) && (defined(__GNUC__) || defined(__clang__))
#  if defined(__HIP_DEVICE_COMPILE__)
        __asm__("" : "+v"(v));
#  else
        AMREXPR_IF_ON_HOST((__asm__("" : "+r"(v));))
#  endif
#endif
        return v;
    }

    template <typename T>
    using fp_uint_t = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;

    //! Exponent bits of T, which are also the bits of +inf.
    template <typename T>
    AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    constexpr fp_uint_t<T> fp_exp_mask () noexcept
    {
        return (sizeof(T) == 8) ? fp_uint_t<T>(0x7ff0'0000'0000'0000ULL)
                                : fp_uint_t<T>(0x7f80'0000U);
    }

    //! |x| as bits: NaN if > fp_exp_mask(), inf if ==, finite if <.
    template <typename T>
    AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    fp_uint_t<T> fp_abs_bits (T x) noexcept
    {
        using U = fp_uint_t<T>;
        U bits;
        std::memcpy(&bits, &x, sizeof(U));
        bits = fp_opaque(bits);
        return bits & (~U(0) >> 1);
    }

    template <typename T>
    constexpr bool is_float_or_double = std::is_same_v<T,float> || std::is_same_v<T,double>;
}

//! Return true if x is NaN.  Works under fast math.
template <typename T>
[[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
bool isnan (T x) noexcept
{
#if defined(AMREXPR_USE_SYCL)
    if constexpr (detail::is_float_or_double<T>) {
        return sycl::isnan(x);
    } else
#elif defined(AMREXPR_MATH_ISNAN_BITS)
    if constexpr (detail::is_float_or_double<T>) {
        return detail::fp_abs_bits(x) > detail::fp_exp_mask<T>();
    } else
#endif
    {
        return std::isnan(x);
    }
}

//! Return true if x is +/-infinity.  Works under fast math.
template <typename T>
[[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
bool isinf (T x) noexcept
{
#if defined(AMREXPR_USE_SYCL)
    if constexpr (detail::is_float_or_double<T>) {
        return sycl::isinf(x);
    } else
#elif defined(AMREXPR_MATH_ISNAN_BITS)
    if constexpr (detail::is_float_or_double<T>) {
        return detail::fp_abs_bits(x) == detail::fp_exp_mask<T>();
    } else
#endif
    {
        return std::isinf(x);
    }
}

//! Return true if x is neither NaN nor infinity.  Works under fast math.
template <typename T>
[[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
bool isfinite (T x) noexcept
{
#if defined(AMREXPR_USE_SYCL)
    if constexpr (detail::is_float_or_double<T>) {
        return sycl::isfinite(x);
    } else
#elif defined(AMREXPR_MATH_ISNAN_BITS)
    if constexpr (detail::is_float_or_double<T>) {
        return detail::fp_abs_bits(x) < detail::fp_exp_mask<T>();
    } else
#endif
    {
        return std::isfinite(x);
    }
}

}

namespace amrexpr {
    using Math::isnan;
    using Math::isinf;
    using Math::isfinite;
}

#endif
