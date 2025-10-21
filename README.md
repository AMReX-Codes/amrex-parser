# amrexpr: AMReX's Mathematical Expression Parser

## Overview

This is a standalone mathematical expression parser library extracted from
the [AMReX](https://github.com/AMReX-Codes/amrex/) software framework. AMReX
is designed for high-performance computing applications that solve partial
differential equations on block-structured adaptive meshes. This library is
for users who wish to utilize the parser functionality without incorporating
the full AMReX framework. It supports both CPU and GPU architectures,
including Nvidia, AMD, and Intel GPUs. While the construction and
initializaion of a `Parser` object are not thread-safe, the evaluation of
parsed expressions is fully thread-safe. The library requires C++17 or
later.

## Features

The parser can be used at runtime to evaluate mathematical expressions given
in the form of string.  It supports `+`, `-`, `*`, `/`, `**` (power), `^`
(power), `sqrt`, `exp`, `log`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`,
`atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `abs`,
`floor`, `ceil`, `fmod`, and `erf`. The minimum and maximum of two numbers
can be computed with `min` and `max`, respectively.  It supports the
Heaviside step function, `heaviside(x1,x2)` that gives `0`, `x2`, `1`, for
`x1 < 0`, `x1 = 0` and `x1 > 0`, respectively.  It supports the Bessel
function of the first kind of order `n` `jn(n,x)`, and the Bessel function
of the second kind of order ``n`` ``yn(n,x)``. Complete elliptic integrals
of the first and second kind, `comp_ellint_1(k)` and `comp_ellint_2(k)`, are
supported.  There is `if(a,b,c)` that gives `b` or `c` depending on the
value of `a`.  A number of comparison operators are supported, including
`<`, `>`, `==`, `!=`, `<=`, and `>=`, and they can be chained. The Boolean
results from comparison can be combined by `and` and `or`, and they hold the
value `1` for true and `0` for false.  The precedence of the operators
follows the convention of the C and C++ programming languages.  Here is an
example of using the parser.

```c++
   #include "amrexpr.hpp"

   Parser parser("if(a<x<b, sin(x)*cos(y)*if(z<0, 1.0, exp(-z)), .3*c**2)");
   parser.setConstant("a", ...);
   parser.setConstant("b", ...);
   parser.setConstant("c", ...);
   parser.registerVariables({"x","y","z"});
   auto f = parser.compile<3>();  // 3 because there are three variables.

   // ParserExecutor<3> f is thread-safe, and can be used in both host and
   // device code. It takes 3 arguments in this example. The parser object
   // must be alive for f to be valid.

   for (int k = 0; ...) {
     for (int j = 0; ...) {
       for (int i = 0; ...) {
         a(i,j,k) = f(i*dx, j*dy, k*dz);
       }
     }
   }
```

Local automatic variables can be defined in the expression.  For example,

```c++
   Parser parser("r2=x*x+y*y; r=sqrt(r2); cos(a+r2)*log(r)");
   parser.setConstant("a", ...);
   parser.registerVariables({"x","y"});
   auto f = parser.compile<2>();  // 2 because there are two variables.
```

Note that an assignment to an automatic variable must be terminated with
``;``, and one should avoid name conflict between the local variables and
the constants set by `setConstant` and the variables registered by
`registerVariables`.

The parser's `operator()` does not throw exceptions because it's meant to be
used on both CPUs and GPUs. However, you could catch run time errors such as
syntax errors during the definition and compilation stages. For example,

```c++
    amrexpr::Parser parser2;
    amrexpr::ParserExecutor<2> exe2;
    try {
        parser2.define("a*x + b*y + b^^3"); // this will cause a syntax error
        parser2.setConstant("a", 4.0);
        parser2.setConstant("b", 2.0);
        parser.registerVariables({"x","y"});
        exe2 = parser.compile<2>(); // 2: two variables
    } catch (std::runtime_error const& e) {
        std::cout << e.what() << "\n";
    }
    if (exe2) {
        std::cout << "There was a syntax error in the expression. How did we come here?\n\n";
    } else {
        std::cout << "exe2 is null as expected\n\n";
    }

    amrexpr::Parser parser3;
    amrexpr::ParserExecutor<2> exe3;
    try {
        parser3.define("a*x + b*y + z");
        parser3.setConstant("a", 4.0);
        parser3.setConstant("b", 2.0);
        parser3.registerVariables({"x","y"}); // forgot about z
        exe3 = parser3.compile<2>(); // 2: two variables
    } catch (std::runtime_error const& e) {
        std::cout << e.what() << "\n";
    }
    if (exe3) {
        std::cout << "There was an unknown symbol in the expression. How did we come here?\n\n";
    } else {
        std::cout << "exe3 is null as expected\n\n";
    }
```

This code block above will result in

```console
syntax error in Parser expression "a*x + b*y + b^^3"
exe2 is null as expected

Unknown variable z in Parser expression "a*x + b*y + z"
exe3 is null as expected
```

## Installation

There two ways to install `amrexpr`. A simple example demonstrating the use
of `libamrexpr` is available at `amrexpr/Tutorials/libamrexpr`.

### Option 1: Using GNU Make

In the `amrexpr` root directory, follow these steps:

```console
$ ./configure
$ make -j8
$ make install
```

To see the list of options, run `./configure -h`. These options allow you to
configure installation directory, choose the compiler, specify GPU backends,
and more.

### Option 2: Using CMake

To build for CPUs, follow these steps in the `amrexpr` root directory:

```console
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<installation_direction>
$ cmake --build build -j 8
$ cmake --install build
```

To build for GPUs, we use the `ENABLE_[CUDA|HIP|SYCL]` option and specify
the GPU architecture in the configure step. For example

```console
# Nvidia GPU w/ compute capability 8.0
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<installation_direction> \
      -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80

# AMD MI250X GPU, gfx90a architecture
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<installation_direction> \
      -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx908

# Intel GPU
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<installation_direction> \
      -DENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=icpx
```

## Copyright Notice

AMReX Copyright (c) 2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

Please see the notices in [NOTICE](NOTICE).

## License

License for AMReX can be found at [LICENSE](LICENSE).
