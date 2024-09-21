#include "amrexpr_Parser.H"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

using namespace amrexpr;

double f (int icase, double x, double y, double z);

bool test_parser(int icase, std::string const& expr)
{
    double x = 1.23, y = 2.34, z = 3.45;
    auto result_native = f(icase, x, y, z);

    Parser parser(expr);
    parser.registerVariables({"x","y","z"});
    auto const exe = parser.compile<3>();
    auto result_parser = exe(x,y,z);

    std::cout << "\ncase " << icase << ": " << expr << "\n";
    parser.printExe();

    return amrexpr::almostEqual(result_native, result_parser, 10);
}

int main (int argc, char* argv[])
{
    std::ifstream ifs("fn.cpp");
    if (!ifs.is_open()) {
        std::cerr << "Failed to open fn.cpp\n";
        return EXIT_FAILURE;
    }

    std::regex case_re("case[[:space:]]+([[:digit:]]+)[[:space:]]*:");
    std::regex expr_re("return[[:space:]]*(.+)[[:space:]]*;");

    int ntests = 0;
    std::vector<std::pair<int,std::string>> failed_tests;

    std::string line;
    std::smatch cm, em;
    while (std::getline(ifs, line)) {
        if (std::regex_search(line, cm, case_re)) {
            int icase = std::stoi(cm[1]);
            if (std::getline(ifs, line)) {
                std::regex_search(line, em, expr_re);
                std::string expr(em[1]);
                ++ntests;
                if (! test_parser(icase, expr)) {
                    failed_tests.push_back(std::make_pair(icase,expr));
                }
            } else {
                std::cerr << "How did this happen? No line after case.\n";
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "\n";
    if (failed_tests.empty()) {
        std::cout << "All " << ntests << " tests passed.\n";
    } else {
        std::cout << failed_tests.size() << " out of " << ntests
                  << " tests failed.\n";
        for (auto const& ie : failed_tests) {
            std::cout << "  case " << ie.first << ": " << ie.second << "\n";
        }
    }

    // return EXIT_SUCCESS;
    return EXIT_FAILURE;
}
