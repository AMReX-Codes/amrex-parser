#include "amrexpr.hpp"
#include <iostream>

int main (int argc, char* argv[])
{
    amrexpr::Parser parser("a*x + b*y");
    parser.setConstant("a", 4.0);
    parser.setConstant("b", 2.0);
    parser.registerVariables({"x","y"});
    auto f = parser.compile<2>(); // 2: two variables
    double x = 10.0;
    double y = 1.0;
    double result = f(x,y);
    std::cout << "f(x,y) = " << parser.expr() << "\n"
              << "f(" << x << "," << y << ") = " << result << "\n";
}
