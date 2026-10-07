#include "amrexpr.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace amrexpr;

namespace {
    int max_stack_size = 0;
    int test_number = 0;
}

template <typename F>
int test1 (std::string const& f,
           std::map<std::string,Real> const& constants,
           std::vector<std::string> const& variables,
           F const& fb, std::array<Real,1> const& lo, std::array<Real,1> const& hi,
           int N, Real reltol, Real abstol)
{
    std::cout << test_number++ << ". Testing \"" << f << "\"   ";

    Parser parser(f);
    for (auto const& kv : constants) {
        parser.setConstant(kv.first, kv.second);
    }
    parser.registerVariables(variables);
    auto const exe = parser.compile<1>();
    max_stack_size = std::max(max_stack_size, parser.maxStackSize());

    std::array<Real,1> dx{(hi[0]-lo[0]) / (N-1)};

    int nfail = 0;
    Real max_relerror = 0.;
    for (int i = 0; i < N; ++i) {
        Real x = lo[0] + i*dx[0];
        Real result = exe(x);
        Real benchmark = fb(x);
        Real abserror = std::abs(result-benchmark);
        Real relerror = abserror / (1.e-50 + std::max(std::abs(result),std::abs(benchmark)));
        if (abserror > abstol && relerror > reltol) {
            std::cout << std::setprecision(17)
                      << "\n    f(" << x << ") = " << result << ", " << benchmark;
            max_relerror = std::max(max_relerror, relerror);
            ++nfail;
        }
    }
    if (nfail > 0) {
        std::cout << "\n    failed " << nfail << " times.  Max rel. error: "
                       << max_relerror << "\n";
        return 1;
    } else {
        std::cout << "    pass\n";
        return 0;
    }
}

template <typename F>
int test3 (std::string const& f,
           std::map<std::string,Real> const& constants,
           std::vector<std::string> const& variables,
           F const& fb, std::array<Real,3> const& lo, std::array<Real,3> const& hi,
           int N, Real reltol, Real abstol)
{
    std::cout << test_number++ << ". Testing \"" << f << "\"   ";

    Parser parser(f);
    for (auto const& kv : constants) {
        parser.setConstant(kv.first, kv.second);
    }
    parser.registerVariables(variables);
    auto const exe = parser.compile<3>();
    max_stack_size = std::max(max_stack_size, parser.maxStackSize());

    std::array<Real,3> dx{(hi[0]-lo[0]) / (N-1),
                          (hi[1]-lo[1]) / (N-1),
                          (hi[2]-lo[2]) / (N-1)};
    int nfail = 0;
    for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
        Real x = lo[0] + i*dx[0];
        Real y = lo[1] + j*dx[1];
        Real z = lo[2] + k*dx[2];
        Real result = exe(x,y,z);
        Real benchmark = fb(x,y,z);
        Real abserror = std::abs(result-benchmark);
        Real relerror = abserror / (1.e-50 + std::max(std::abs(result),std::abs(benchmark)));
        if (abserror > abstol && relerror > reltol) {
            std::cout << "    f(" << x << "," << y << "," << z << ") = " << result << ", "
                           << benchmark << "\n";
            ++nfail;
        }
    }}}
    if (nfail > 0) {
        std::cout << "    failed " << nfail << " times\n";
        return 1;
    } else {
        std::cout << "    pass\n";
        return 0;
    }
}

template <typename F>
int test4 (std::string const& f,
           std::map<std::string,Real> const& constants,
           std::vector<std::string> const& variables,
           F const& fb, std::array<Real,4> const& lo, std::array<Real,4> const& hi,
           int N, Real reltol, Real abstol)
{
    std::cout << test_number++ << ". Testing \"" << f << "\"   ";

    Parser parser(f);
    for (auto const& kv : constants) {
        parser.setConstant(kv.first, kv.second);
    }
    parser.registerVariables(variables);
    auto const exe = parser.compile<4>();
    max_stack_size = std::max(max_stack_size, parser.maxStackSize());

    std::array<Real,4> dx{(hi[0]-lo[0]) / (N-1),
                          (hi[1]-lo[1]) / (N-1),
                          (hi[2]-lo[2]) / (N-1),
                          (hi[3]-lo[3]) / (N-1)};
    int nfail = 0;
    for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
    for (int m = 0; m < N; ++m) {
        Real x = lo[0] + i*dx[0];
        Real y = lo[1] + j*dx[1];
        Real z = lo[2] + k*dx[2];
        Real t = lo[3] + m*dx[3];
        Real result = exe(x,y,z,t);
        Real benchmark = fb(x,y,z,t);
        Real abserror = std::abs(result-benchmark);
        Real relerror = abserror / (1.e-50 + std::max(std::abs(result),std::abs(benchmark)));
        if (abserror > abstol && relerror > reltol) {
            std::cout << "    f(" << x << "," << y << "," << z << "," << t << ") = " << result << ", "
                           << benchmark << "\n";
            ++nfail;
        }
    }}}}
    if (nfail > 0) {
        std::cout << "    failed " << nfail << " times\n";
        return 1;
    } else {
        std::cout << "    pass\n";
        return 0;
    }
}

int test_concurrent_parser_construction ()
{
    std::cout << test_number++ << ". Testing concurrent Parser construction   ";

    constexpr int nthreads = 2;
    constexpr int niters = 64;
    std::vector<int> nfail(nthreads, 0);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    for (int tid = 0; tid < nthreads; ++tid) {
        threads.emplace_back([tid, &nfail] ()
        {
            int local_nfail = 0;
            for (int iter = 0; iter < niters; ++iter) {
                try {
                    {
                        Parser parser("a*x + b");
                        auto const a = static_cast<double>(tid+1);
                        auto const b = 0.25*static_cast<double>(iter+1);
                        auto const x = static_cast<double>((iter % 7) - 3);
                        parser.setConstant("a", a);
                        parser.setConstant("b", b);
                        parser.registerVariables({"x"});
                        auto const exe = parser.compileHost<1>();
                        if (std::abs(exe(x) - (a*x + b)) > 1.e-12) {
                            ++local_nfail;
                        }
                    }
                    {
                        Parser parser("if(x > threshold, x*x + c, c-x)");
                        auto const threshold = static_cast<double>((tid % 3) - 1);
                        auto const c = 0.125*static_cast<double>(iter+1);
                        auto const x = static_cast<double>((iter % 5) - 2);
                        parser.setConstant("threshold", threshold);
                        parser.setConstant("c", c);
                        parser.registerVariables({"x"});
                        auto const exe = parser.compileHost<1>();
                        auto const expected = (x > threshold) ? x*x + c : c - x;
                        if (std::abs(exe(x) - expected) > 1.e-12) {
                            ++local_nfail;
                        }
                    }
                } catch (...) {
                    ++local_nfail;
                }
            }
            nfail[tid] = local_nfail;
        });
    }

    int total_nfail = 0;
    for (auto& thread : threads) {
        thread.join();
    }
    for (auto n : nfail) {
        total_nfail += n;
    }

    if (total_nfail > 0) {
        std::cout << "    failed " << total_nfail << " times\n";
        return 1;
    } else {
        std::cout << "    pass\n";
        return 0;
    }
}

int main (int argc, char* argv[])
{
    amrexpr::ignore_unused(argc, argv);

    int nerror = 0;
    {
        std::cout << "\n";
        nerror += test3("if( ((z-zc)*(z-zc)+(y-yc)*(y-yc)+(x-xc)*(x-xc))^(0.5) < (r_star-dR), 0.0, if(((z-zc)*(z-zc)+(y-yc)*(y-yc)+(x-xc)*(x-xc))^(0.5) <= r_star, dens, 0.0))",
                    {{"xc", 0.1}, {"yc", -1.0}, {"zc", 0.2}, {"r_star", 0.73}, {"dR", 0.57}, {"dens", 12.}},
                    {"x","y","z"},
                    [=] (Real x, Real y, Real z) -> Real {
                        Real xc=0.1, yc=-1.0, zc=0.2, r_star=0.73, dR=0.57, dens=12.;
                        Real r = std::sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc));
                        if (r >= r_star-dR && r <= r_star) {
                            return dens;
                        } else {
                            return 0.0;
                        }
                    },
                    {-1., -1., -1.0}, {1.0, 1.0, 1.0}, 100,
                    1.e-12, 1.e-15);

        nerror += test3("r=sqrt((z-zc)*(z-zc)+(y-yc)*(y-yc)+(x-xc)*(x-xc)); if(r < (r_star-dR), 0.0, if(r <= r_star, dens, 0.0))",
                        {{"xc", 0.1}, {"yc", -1.0}, {"zc", 0.2}, {"r_star", 0.73}, {"dR", 0.57}, {"dens", 12.}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real xc=0.1, yc=-1.0, zc=0.2, r_star=0.73, dR=0.57, dens=12.;
                            Real r = std::sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc));
                            if (r >= r_star-dR && r <= r_star) {
                                return dens;
                            } else {
                                return 0.0;
                            }
                        },
                        {-1., -1., -1.0}, {1.0, 1.0, 1.0}, 100,
                        1.e-12, 1.e-15);

        nerror += test3("r2=(z-zc)*(z-zc)+(y-yc)*(y-yc)+(x-xc)*(x-xc); r=sqrt(r2); if(r < (r_star-dR), 0.0, if(r <= r_star, dens, 0.0))",
                        {{"xc", 0.1}, {"yc", -1.0}, {"zc", 0.2}, {"r_star", 0.73}, {"dR", 0.57}, {"dens", 12.}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real xc=0.1, yc=-1.0, zc=0.2, r_star=0.73, dR=0.57, dens=12.;
                            Real r = std::sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc));
                            if (r >= r_star-dR && r <= r_star) {
                                return dens;
                            } else {
                                return 0.0;
                            }
                        },
                        {-1., -1., -1.0}, {1.0, 1.0, 1.0}, 100,
                        1.e-12, 1.e-15);

        nerror += test3("( ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<=r_star) * ((( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>=(r_star-dR)) )*dens",
                        {{"xc", 0.1}, {"yc", -1.0}, {"zc", 0.2}, {"r_star", 0.73}, {"dR", 0.57}, {"dens", 12.}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real xc=0.1, yc=-1.0, zc=0.2, r_star=0.73, dR=0.57, dens=12.;
                            Real r = std::sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc));
                            if (r >= r_star-dR && r <= r_star) {
                                return dens;
                            } else {
                                return 0.0;
                            }
                        },
                        {-1., -1., -1.0}, {1.0, 1.0, 1.0}, 100,
                        1.e-12, 1.e-15);

        nerror += test4("( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))>=r_star)*(-( (t<to)*(t/to)*omega + (t>=to)*omega )*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-( ( (t<to)*(t/to)*omega + (t>=to)*omega)  *(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (y-yc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))",
                        {{"xc", 0.1}, {"yc", -1.0}, {"zc", 0.2}, {"to", 3.}, {"omega", 0.33}, {"c", 30.}, {"r_star", 0.75}},
                        {"x","y","z","t"},
                        [=] (Real x, Real y, Real z, Real t) -> Real {
                            Real xc=0.1, yc=-1.0, zc=0.2, to=3., omega=0.33, c=30., r_star=0.75;
                            Real r = std::sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc));
                            if (r >= r_star) {
                                Real tomega = (t>=to) ? omega : omega*(t/to);
                                Real r2d = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc));
                                return -tomega * r / std::sqrt(1.0-(tomega*r2d/c)*(tomega*r2d/c)) * (y-yc)/r;
                            } else {
                                return 0.0;
                            }
                        },
                        {-1., -1., -1.0, 0.0}, {1.0, 1.0, 1.0, 10}, 30,
                        1.e-12, 1.e-15);

        nerror += test4("r=sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc)); tomega=if(t>=to, omega, omega*(t/to)); r2d=sqrt((y-yc)*(y-yc) + (x-xc)*(x-xc)); (r>=r_star)*(-tomega*r/(1.0-((tomega*r2d/c)^2))^0.5 * (y-yc)/r)",
                        {{"xc", 0.1}, {"yc", -1.0}, {"zc", 0.2}, {"to", 3.}, {"omega", 0.33}, {"c", 30.}, {"r_star", 0.75}},
                        {"x","y","z","t"},
                        [=] (Real x, Real y, Real z, Real t) -> Real {
                            Real xc=0.1, yc=-1.0, zc=0.2, to=3., omega=0.33, c=30., r_star=0.75;
                            Real r = std::sqrt((z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc));
                            if (r >= r_star) {
                                Real tomega = (t>=to) ? omega : omega*(t/to);
                                Real r2d = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc));
                                return -tomega * r / std::sqrt(1.0-(tomega*r2d/c)*(tomega*r2d/c)) * (y-yc)/r;
                            } else {
                                return 0.0;
                            }
                        },
                        {-1., -1., -1.0, 0.0}, {1.0, 1.0, 1.0, 10}, 30,
                        1.e-12, 1.e-15);

        nerror += test3("cos(m * pi / Lx * (x - Lx / 2)) * cos(n * pi / Ly * (y - Ly / 2)) * sin(p * pi / Lz * (z - Lz / 2))*mu_0*(x>-0.5)*(x<0.5)*(y>-0.5)*(y<0.5)*(z>-0.5)*(z<0.5)",
                        {{"m", 0.0}, {"n", 1.0}, {"pi", 3.14}, {"p", 1.0}, {"Lx", 1.}, {"Ly", 1.}, {"Lz", 1.}, {"mu_0", 1.27e-6}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real m=0.0,n=1.0,pi=3.14,p=1.0,Lx=1.,Ly=1.,Lz=1.,mu_0=1.27e-6;
                            if ((x>-0.5) && (x<0.5) && (y>-0.5) && (y<0.5) && (z>-0.5) &&(z<0.5)) {
                                return std::cos(m * pi / Lx * (x - Lx / 2)) * std::cos(n * pi / Ly * (y - Ly / 2)) * std::sin(p * pi / Lz * (z - Lz / 2))*mu_0;
                            } else {
                                return 0.0;
                            }
                        },
                        {-0.8, -0.8, -0.8}, {0.8, 0.8, 0.8}, 100,
                        1.e-12, 1.e-15);

        nerror += test3("if ((x>-0.5) and (x<0.5) and (y>-0.5) and (y<0.5) and (z>-0.5) and (z<0.5), cos(m * pi / Lx * (x - Lx / 2)) * cos(n * pi / Ly * (y - Ly / 2)) * sin(p * pi / Lz * (z - Lz / 2))*mu_0*(x>-0.5)*(x<0.5)*(y>-0.5)*(y<0.5)*(z>-0.5)*(z<0.5), 0)",
                        {{"m", 0.0}, {"n", 1.0}, {"pi", 3.14}, {"p", 1.0}, {"Lx", 1.}, {"Ly", 1.}, {"Lz", 1.}, {"mu_0", 1.27e-6}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real m=0.0,n=1.0,pi=3.14,p=1.0,Lx=1.,Ly=1.,Lz=1.,mu_0=1.27e-6;
                            if ((x>-0.5) && (x<0.5) && (y>-0.5) && (y<0.5) && (z>-0.5) &&(z<0.5)) {
                                return std::cos(m * pi / Lx * (x - Lx / 2)) * std::cos(n * pi / Ly * (y - Ly / 2)) * std::sin(p * pi / Lz * (z - Lz / 2))*mu_0;
                            } else {
                                return 0.0;
                            }
                        },
                        {-0.8, -0.8, -0.8}, {0.8, 0.8, 0.8}, 100,
                        1.e-12, 1.e-15);

        nerror += test3("if ((-0.5 < x < 0.5) and (-0.5< (x+y) <0.5) and (-0.5<z<0.5), cos(m * pi / Lx * (x - Lx / 2)) * cos(n * pi / Ly * (y - Ly / 2)) * sin(p * pi / Lz * (z - Lz / 2))*mu_0, 0)",
                        {{"m", 0.0}, {"n", 1.0}, {"pi", 3.14}, {"p", 1.0}, {"Lx", 1.}, {"Ly", 1.}, {"Lz", 1.}, {"mu_0", 1.27e-6}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real m=0.0,n=1.0,pi=3.14,p=1.0,Lx=1.,Ly=1.,Lz=1.,mu_0=1.27e-6;
                            if ((x>-0.5) && (x<0.5) && ((x+y)>-0.5) && ((x+y)<0.5) && (z>-0.5) &&(z<0.5)) {
                                return std::cos(m * pi / Lx * (x - Lx / 2)) * std::cos(n * pi / Ly * (y - Ly / 2)) * std::sin(p * pi / Lz * (z - Lz / 2))*mu_0;
                            } else {
                                return 0.0;
                            }
                        },
                        {-0.8, -0.8, -0.8}, {0.8, 0.8, 0.8}, 100,
                        1.e-12, 1.e-15);

        nerror += test3("cos(m * pi / Lx * (x - Lx / 2)) * cos(n * pi / Ly * (y - Ly / 2)) * sin(p * pi / Lz * (z - Lz / 2))*mu_0*(0.5>x>y>z>(x+z)>-0.5)",
                        {{"m", 0.0}, {"n", 1.0}, {"pi", 3.14}, {"p", 1.0}, {"Lx", 1.}, {"Ly", 1.}, {"Lz", 1.}, {"mu_0", 1.27e-6}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real m=0.0,n=1.0,pi=3.14,p=1.0,Lx=1.,Ly=1.,Lz=1.,mu_0=1.27e-6;
                            if ((0.5>x) && (x>y) && (y>z) && (z>(x+z)) && ((x+z)>-0.5)) {
                                return std::cos(m * pi / Lx * (x - Lx / 2)) * std::cos(n * pi / Ly * (y - Ly / 2)) * std::sin(p * pi / Lz * (z - Lz / 2))*mu_0;
                            } else {
                                return 0.0;
                            }
                        },
                        {-0.8, -0.8, -0.8}, {0.8, 0.8, 0.8}, 100,
                        1.e-12, 1.e-15);

        nerror += test3("2.*sqrt(2.)+sqrt(-log(x))*cos(2*pi*z)",
                        {{"pi", 3.14}},
                        {"x","y","z"},
                        [=] (Real x, Real, Real z) -> Real {
                            Real pi = 3.14;
                            return 2.*std::sqrt(2.)+std::sqrt(-std::log(x))*std::cos(2*pi*z);
                        },
                        {0.5, 0.8, 0.3}, {16, 16, 16}, 100,
                        1.e-12, 1.e-15);

        nerror += test1("nc*n0*(if(abs(z)<=r0, 1.0, if(abs(z)<r0+Lcut, exp((-abs(z)+r0)/L), 0.0)))",
                        {{"nc",1.742e27},{"n0",30.},{"r0",2.5e-6},{"Lcut",2.e-6},{"L",0.05e-6}},
                        {"z"},
                        [=] (Real z) -> Real {
                            Real nc=1.742e27, n0=30., r0=2.5e-6, Lcut=2.e-6, L=0.05e-6;
                            if (std::abs(z) <= r0) {
                                return nc*n0;
                            } else if (std::abs(z) < r0+Lcut) {
                                return nc*n0*std::exp((-std::abs(z)+r0)/L);
                            } else {
                                return 0.0;
                            }
                        },
                        {-5.e-6}, {25.e-6}, 10000,
                        1.e-12, 1.e-15);

        nerror += test1("(z<lramp)*0.5*(1-cos(pi*z/lramp))*dens+(z>lramp)*dens",
                        {{"lramp",8.e-3},{"pi",3.14},{"dens",1.e23}},
                        {"z"},
                        [=] (Real z) -> Real {
                            Real lramp=8.e-3, pi=3.14, dens=1.e23;
                            if (z < lramp) {
                                return 0.5*(1-std::cos(pi*z/lramp))*dens;
                            } else {
                                return dens;
                            }
                        },
                        {-149.e-6}, {1.e-6}, 1000,
                        1.e-12, 1.e-15);

        nerror += test1("if(z<lramp, 0.5*(1-cos(pi*z/lramp))*dens, dens)",
                        {{"lramp",8.e-3},{"pi",3.14},{"dens",1.e23}},
                        {"z"},
                        [=] (Real z) -> Real {
                            Real lramp=8.e-3, pi=3.14, dens=1.e23;
                            if (z < lramp) {
                                //return 0.5*(1-std::cos(pi*z/lramp))*dens;
                                return 0.5*dens-0.5*dens*std::cos(pi*z/lramp);
                            } else {
                                return dens;
                            }
                        },
                        {-149.e-6}, {1.e-6}, 1000,
                        1.e-12, 1.e-15);

        nerror += test1("if(z<zp, nc*exp((z-zc)/lgrad), if(z<=zp2, 2.*nc, nc*exp(-(z-zc2)/lgrad)))",
                        {{"zc",20.e-6},{"zp",20.05545177444479562e-6},{"nc",1.74e27},{"lgrad",0.08e-6},{"zp2",24.e-6},{"zc2",24.05545177444479562e6}},
                        {"z"},
                        [=] (Real z) -> Real {
                            Real zc=20.e-6, zp=20.05545177444479562e-6, nc=1.74e27, lgrad=0.08e-6, zp2=24.e-6, zc2=24.05545177444479562e6;
                            if (z < zp) {
                                return nc*std::exp((z-zc)/lgrad);
                            } else if (z <= zp2) {
                                return 2.*nc;
                            } else {
                                return nc*exp(-(z-zc2)/lgrad);
                            }
                        },
                        {0.}, {100.e-6}, 1000,
                        1.e-12, 1.e-15);

        // f(x)*f(x) => square, and b + a*x => fma
        nerror += test1("sin(x)*sin(x)", {}, {"x"},
                        [=] (Real x) -> Real { return std::sin(x)*std::sin(x); },
                        {-3.0}, {3.0}, 101, 1.e-12, 1.e-15);
        nerror += test1("1.5 + 2.5*x", {}, {"x"},
                        [=] (Real x) -> Real { return 1.5 + 2.5*x; },
                        {-3.0}, {3.0}, 101, 1.e-12, 1.e-15);

        // CR and LF are stripped.
        nerror += test1("x\r\n + 1\r\n", {}, {"x"},
                        [=] (Real x) -> Real { return x + 1.0; },
                        {-3.0}, {3.0}, 7, 1.e-12, 1.e-15);

        {   // setConstant and registerVariables are refused after compile().
            std::cout << test_number++ << ". Testing setConstant/registerVariables after compile   ";
            Parser parser("a*x");
            parser.setConstant("a", 2.0);
            parser.registerVariables({"x"});
            auto exe = parser.compile<1>();
            int ncaught = 0;
            try { parser.setConstant("a", 3.0); } catch (std::runtime_error const&) { ++ncaught; }
            try { parser.registerVariables({"x"}); } catch (std::runtime_error const&) { ++ncaught; }
            if (ncaught == 2 && exe(1.5) == 3.0) {
                std::cout << "    pass\n";
            } else {
                std::cout << "    failed\n";
                ++nerror;
            }
        }

        nerror += test3("epsilon/kp*2*x/w0**2*exp(-(x**2+y**2)/w0**2)*sin(k0*z)",
                        {{"epsilon",0.01},{"kp",3.5},{"w0",5.e-6},{"k0",3.e5}},
                        {"x","y","z"},
                        [=] (Real x, Real y, Real z) -> Real {
                            Real epsilon=0.01, kp=3.5, w0=5.e-6, k0=3.e5;
                            return epsilon/kp*2*x/(w0*w0)*std::exp(-(x*x+y*y)/(w0*w0))*sin(k0*z);
                        },
                        {0.e-6, 0.0, -20.e-6}, {20.e-6, 1.e-10, 20.e-6}, 100,
                        1.e-12, 1.e-15);

        {   // An expression that optimizes to nothing must be rejected.
            std::cout << test_number++ << ". Testing \"a=1; b=2;\"   ";
            try {
                Parser parser("a=1; b=2;");
                std::cout << "    failed: no exception\n";
                ++nerror;
            } catch (std::runtime_error const& e) {
                std::cout << "    pass\n";
            }
        }

        {   // The same via setConstant. The Parser must then be undefined,
            // not left with an empty syntax tree.
            std::cout << test_number++ << ". Testing \"a=c;\" with c set to a constant   ";
            Parser parser("a=c;");
            bool caught = false;
            try {
                parser.setConstant("c", 2.0);
            } catch (std::runtime_error const& e) {
                caught = true;
            }
            if (caught && !parser && parser.symbols().empty() && parser.depth() == 0) {
                std::cout << "    pass\n";
            } else {
                std::cout << "    failed\n";
                ++nerror;
            }
        }

        nerror += test_concurrent_parser_construction();

        {   // A failed compile must not leave a stale executor behind.
            std::cout << test_number++ << ". Testing recovery after a failed compile   ";
            Parser parser("x+y");
            parser.registerVariables({"x"});
            bool ok = false;
            try {
                auto exe = parser.compile<1>(); // y is unknown
                amrexpr::ignore_unused(exe);
            } catch (std::runtime_error const&) {
                ok = true;
            }
            if (ok) {
                try {
                    parser.setConstant("y", 2.0); // must not be refused
                    auto exe = parser.compile<1>();
                    ok = (exe(3.0) == 5.0);
                } catch (std::runtime_error const& e) {
                    std::cout << "\n    " << e.what();
                    ok = false;
                }
            }
            if (ok) {
                std::cout << "    pass\n";
            } else {
                std::cout << "    failed\n";
                ++nerror;
            }
        }

#if !(defined(AMREXPR_USE_SYCL) && !(defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_CLANG_COMPILER)))
        for (int n : {-5, -2, -1, 0, 1, 2, 5}) {
            nerror += test1("jn(" + std::to_string(n) + ",x)", {}, {"x"},
                            [=] (double x) -> double {
#if defined(_WIN32) && defined(__MINGW32__)
                                int const m = n < 0 ? -n : n;
                                double const xa = x < 0.0 ? -x : x;
                                double r = std::cyl_bessel_j(double(m), xa);
                                if ((m % 2 != 0) && ((n < 0) != (x < 0.0))) {
                                    r = -r;
                                }
                                return r;
#elif defined(_WIN32)
                                return ::_jn(n, x);
#else
                                return ::jn(n, x);
#endif
                            },
                            {-5.0}, {5.0}, 100, 1.e-12, 1.e-14);

            nerror += test1("yn(" + std::to_string(n) + ",x)", {}, {"x"},
                            [=] (double x) -> double {
#if defined(_WIN32) && defined(__MINGW32__)
                                int const m = n < 0 ? -n : n;
                                double r = std::cyl_neumann(double(m), x);
                                if ((m % 2 != 0) && (n < 0)) {
                                    r = -r;
                                }
                                return r;
#elif defined(_WIN32)
                                return ::_yn(n, x);
#else
                                return ::yn(n, x);
#endif
                            },
                            {0.5}, {5.0}, 100, 1.e-12, 1.e-14);
        }
#endif

        // Edge cases where the optimizer must agree with the runtime evaluator.

        // pow(pow(x,m),n) may only be merged for integer m and n.
        nerror += test1("(x**2)**0.5", {}, {"x"},
                        [=] (double x) -> double { return std::pow(std::pow(x,2.0),0.5); },
                        {-3.0}, {3.0}, 101, 1.e-12, 1.e-15);
        nerror += test1("(x**2)**1.5", {}, {"x"},
                        [=] (double x) -> double { return std::pow(std::pow(x,2.0),1.5); },
                        {-3.0}, {3.0}, 101, 1.e-12, 1.e-15);
        nerror += test1("((x-1)*(x-1))**0.5", {}, {"x"},
                        [=] (double x) -> double { return std::pow((x-1.0)*(x-1.0),0.5); },
                        {-3.0}, {3.0}, 101, 1.e-12, 1.e-15);
        // Integer exponents still merge.
        nerror += test1("(x**2)**3", {}, {"x"},
                        [=] (double x) -> double { return std::pow(std::pow(x,2.0),3.0); },
                        {-3.0}, {3.0}, 101, 1.e-12, 1.e-15);

        // Integer exponents too large for int must not take the powi path.
        // Bases near 1 keep the results away from 0 and inf.
        nerror += test1("x**3e9", {}, {"x"},
                        [=] (double x) -> double { return std::pow(x,3.e9); },
                        {1.0-1.e-10}, {1.0+1.e-10}, 5, 1.e-12, 1.e-15);
        nerror += test1("x**-2147483648", {}, {"x"},
                        [=] (double x) -> double { return std::pow(x,-2147483648.); },
                        {1.0-1.e-10}, {1.0+1.e-10}, 5, 1.e-12, 1.e-15);
        // This one fits in int and takes the powi path, whose repeated
        // squaring loses about 31 bits for an exponent this large.
        nerror += test1("x**-2147483647", {}, {"x"},
                        [=] (double x) -> double { return std::pow(x,-2147483647.); },
                        {1.0-1.e-10}, {1.0+1.e-10}, 5, 1.e-8, 1.e-15);

        // A // comment must end at its own line.
        nerror += test1("x + 1 // add one\n + 2*x // add 2x\r\n - 3", {}, {"x"},
                        [=] (double x) -> double { return 3.0*x - 2.0; },
                        {-1.0}, {1.0}, 5, 1.e-12, 1.e-15);

        // pow with a constant zero base: std::pow(0,0) is 1 and pow(0,-1) is inf.
        nerror += test1("c**x", {{"c",0.0}}, {"x"},
                        [=] (double x) -> double { return std::pow(0.0,x); },
                        {0.0}, {4.0}, 5, 1.e-12, 1.e-15);
        nerror += test1("0**x", {}, {"x"},
                        [=] (double x) -> double { return std::pow(0.0,x); },
                        {0.0}, {4.0}, 5, 1.e-12, 1.e-15);

        // and/or must return 1 or 0, not the operand.
        nerror += test1("x and 1", {}, {"x"},
                        [=] (double x) -> double { return (x != 0.0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);
        nerror += test1("1 and x", {}, {"x"},
                        [=] (double x) -> double { return (x != 0.0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);
        nerror += test1("(x and 1)+1", {}, {"x"},
                        [=] (double x) -> double { return ((x != 0.0) ? 1.0 : 0.0) + 1.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);
        nerror += test1("x or 0", {}, {"x"},
                        [=] (double x) -> double { return (x != 0.0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);
        nerror += test1("0 or x", {}, {"x"},
                        [=] (double x) -> double { return (x != 0.0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);
        nerror += test1("x and c", {{"c",1.0}}, {"x"},
                        [=] (double x) -> double { return (x != 0.0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);
        // An operand that is already 0 or 1 needs no extra operation.
        nerror += test1("(x>0) and 1", {}, {"x"},
                        [=] (double x) -> double { return (x > 0.0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 5, 1.e-12, 1.e-15);

        // A parenthesized comparison must not be extended into a chain.
        nerror += test1("(x<1)==0", {}, {"x"},
                        [=] (double x) -> double { return ((x < 1.0) == 0) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 9, 1.e-12, 1.e-15);
        nerror += test1("(x<1)<0.5", {}, {"x"},
                        [=] (double x) -> double { return (double(x < 1.0) < 0.5) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 9, 1.e-12, 1.e-15);
        nerror += test1("x<1<0.5", {}, {"x"},
                        [=] (double x) -> double { return (x < 1.0 && 1.0 < 0.5) ? 1.0 : 0.0; },
                        {-2.0}, {2.0}, 9, 1.e-12, 1.e-15);

        {   // Re-registering must forget the variables it drops, even when the
            // syntax tree is shared with a copy of the Parser.
            std::cout << test_number++ << ". Testing Parser re-registration\n";
            int const nerror0 = nerror;
            auto expect_unknown = [&] (Parser const& p) -> bool
            {
                try {
                    Parser q = p; // NOLINT(performance-unnecessary-copy-initialization)
                    auto exe = q.compile<1>();
                    auto r = exe(2.0);
                    amrexpr::ignore_unused(r);
                    return false;
                } catch (std::runtime_error const& e) {
                    std::cout << "    Expected error: " << e.what() << '\n';
                    return true;
                }
            };
            {
                Parser p("x+y");
                p.registerVariables({"x","y"});
                p.registerVariables({"y"});
                if (!expect_unknown(p)) { ++nerror; }
            }
            {   // The copy registers, so the original's stale binding must go.
                Parser p("x+y");
                Parser q = p;
                p.registerVariables({"x","y"});
                q.registerVariables({"y"});
                if (!expect_unknown(q)) { ++nerror; }
            }
            {   // Reordering is still allowed.
                Parser p("x-y");
                p.registerVariables({"x","y"});
                p.registerVariables({"y","x"});
                auto exe = p.compile<2>();
                if (exe(3.0,10.0) != 7.0) { ++nerror; } // y=3, x=10
            }
            std::cout << ((nerror == nerror0) ? "    pass\n" : "    failed\n");
        }
    }

    std::cout << "\nMax stack size is " << max_stack_size << "\n";
    if (nerror > 0) {
        std::cout << nerror << " tests failed\n\n";
        return EXIT_FAILURE;
    } else {
        std::cout << "All tests passed\n\n";
        return EXIT_SUCCESS;
    }
}
