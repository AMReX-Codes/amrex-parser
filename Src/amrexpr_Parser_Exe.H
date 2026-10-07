#ifndef AMREXPR_PARSER_EXE_H_
#define AMREXPR_PARSER_EXE_H_
#include "amrexpr_Config.H"

/**
 * \file amrexpr_Parser_Exe.H
 * \brief Bytecode instruction formats and helpers for the runtime parser.
 */

#include "amrexpr_BLassert.H"
#include "amrexpr_Parser_Y.H"

#include <limits>
#include <vector>

#ifndef AMREXPR_PARSER_STACK_SIZE
#define AMREXPR_PARSER_STACK_SIZE 16
#endif

// In host-only builds with GCC or Clang, parser_exe_eval uses threaded
// dispatch (computed goto), which is faster than a switch and makes the
// speed less dependent on code layout.  Device compilers and MSVC use the
// switch.  Define AMREXPR_PARSER_THREADED_DISPATCH=0 to use the switch
// everywhere.
#ifndef AMREXPR_PARSER_THREADED_DISPATCH
#if !defined(AMREXPR_USE_GPU) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define AMREXPR_PARSER_THREADED_DISPATCH 1
#else
#define AMREXPR_PARSER_THREADED_DISPATCH 0
#endif
#endif

// Threaded dispatch is used when the bytecode is longer than this (bytes).
#ifndef AMREXPR_PARSER_THREADED_MIN_SIZE
#define AMREXPR_PARSER_THREADED_MIN_SIZE 48
#endif

#define AMREXPR_PARSER_LOCAL_IDX0 1000
// Local variable k is stack entry k, which lives in pdata[k+1] once it
// has been spilled from the cached top (see parser_exe_eval).
#define AMREXPR_PARSER_GET_DATA(i) (((i) < AMREXPR_PARSER_LOCAL_IDX0) ? x[i] : \
    pdata[(i)-(AMREXPR_PARSER_LOCAL_IDX0-1)])

namespace amrexpr {

// N: node
// P: pointer offset
// V: value (i.e., double literal)

// Opcodes.  The threaded dispatch table in amrexpr_Parser_Exe_Body.H is
// generated from this list, so the two stay in sync.
#define AMREXPR_PARSER_EXE_TYPES(X) \
    X(NULL) X(NUMBER) X(SYMBOL) X(ADD) X(SUB_F) X(SUB_B) X(MUL) \
    X(DIV_F) X(DIV_B) X(F1) X(F2_F) X(F2_B) X(ADD_VP) X(SUB_VP) \
    X(MUL_VP) X(DIV_VP) X(ADD_PP) X(SUB_PP) X(MUL_PP) X(DIV_PP) \
    X(ADD_VN) X(SUB_VN) X(MUL_VN) X(DIV_VN) X(ADD_PN) X(SUB_PN) \
    X(MUL_PN) X(DIV_PN) X(SQUARE) X(POWI) X(FMA_VPV) X(IF) X(JUMP) \
    X(F1_P) X(SQUARE_P) X(POWI_P) X(F2_PP) X(F2_PV) X(F2_VP)

enum parser_exe_t {
#define AMREXPR_PARSER_EXE_ENUM(name) PARSER_EXE_##name,
    AMREXPR_PARSER_EXE_TYPES(AMREXPR_PARSER_EXE_ENUM)
#undef AMREXPR_PARSER_EXE_ENUM
};

struct alignas(8) ParserExeNull {
    enum parser_exe_t type = PARSER_EXE_NULL;
};

struct alignas(8) ParserExeNumber {
    enum parser_exe_t type = PARSER_EXE_NUMBER;
    double v;
};

struct alignas(8) ParserExeSymbol {
    enum parser_exe_t type = PARSER_EXE_SYMBOL;
    int i;
};

struct alignas(8) ParserExeADD {
    enum parser_exe_t type = PARSER_EXE_ADD;
};

struct alignas(8) ParserExeSUB_F {
    enum parser_exe_t type = PARSER_EXE_SUB_F;
};

struct alignas(8) ParserExeSUB_B {
    enum parser_exe_t type = PARSER_EXE_SUB_B;
};

struct alignas(8) ParserExeMUL {
    enum parser_exe_t type = PARSER_EXE_MUL;
};

struct alignas(8) ParserExeDIV_F {
    enum parser_exe_t type = PARSER_EXE_DIV_F;
};

struct alignas(8) ParserExeDIV_B {
    enum parser_exe_t type = PARSER_EXE_DIV_B;
};

struct alignas(8) ParserExeF1 {
    enum parser_exe_t type = PARSER_EXE_F1;
    parser_f1_t ftype;
};

struct alignas(8) ParserExeF2_F {
    enum parser_exe_t type = PARSER_EXE_F2_F;
    parser_f2_t ftype;
};

struct alignas(8) ParserExeF2_B {
    enum parser_exe_t type = PARSER_EXE_F2_B;
    parser_f2_t ftype;
};

struct alignas(8) ParserExeADD_VP {
    enum parser_exe_t type = PARSER_EXE_ADD_VP;
    int i;
    double v;
};

struct alignas(8) ParserExeSUB_VP {
    enum parser_exe_t type = PARSER_EXE_SUB_VP;
    int i;
    double v;
};

struct alignas(8) ParserExeMUL_VP {
    enum parser_exe_t type = PARSER_EXE_MUL_VP;
    int i;
    double v;
};

struct alignas(8) ParserExeDIV_VP {
    enum parser_exe_t type = PARSER_EXE_DIV_VP;
    int i;
    double v;
};

struct alignas(8) ParserExeADD_PP {
    enum parser_exe_t type = PARSER_EXE_ADD_PP;
    int i1;
    int i2;
};

struct alignas(8) ParserExeSUB_PP {
    enum parser_exe_t type = PARSER_EXE_SUB_PP;
    int i1;
    int i2;
};

struct alignas(8) ParserExeMUL_PP {
    enum parser_exe_t type = PARSER_EXE_MUL_PP;
    int i1;
    int i2;
};

struct alignas(8) ParserExeDIV_PP {
    enum parser_exe_t type = PARSER_EXE_DIV_PP;
    int i1;
    int i2;
};

struct alignas(8) ParserExeADD_VN {
    enum parser_exe_t type = PARSER_EXE_ADD_VN;
    double v;
};

struct alignas(8) ParserExeSUB_VN {
    enum parser_exe_t type = PARSER_EXE_SUB_VN;
    double v;
};

struct alignas(8) ParserExeMUL_VN {
    enum parser_exe_t type = PARSER_EXE_MUL_VN;
    double v;
};

struct alignas(8) ParserExeDIV_VN {
    enum parser_exe_t type = PARSER_EXE_DIV_VN;
    double v;
};

struct alignas(8) ParserExeADD_PN {
    enum parser_exe_t type = PARSER_EXE_ADD_PN;
    int i;
};

struct alignas(8) ParserExeSUB_PN {
    enum parser_exe_t type = PARSER_EXE_SUB_PN;
    int i;
    bool reverse;
};

struct alignas(8) ParserExeMUL_PN {
    enum parser_exe_t type = PARSER_EXE_MUL_PN;
    int i;
};

struct alignas(8) ParserExeDIV_PN {
    enum parser_exe_t type = PARSER_EXE_DIV_PN;
    int i;
    bool reverse;
};

struct alignas(8) ParserExeSquare {
    enum parser_exe_t type = PARSER_EXE_SQUARE;
};

struct alignas(8) ParserExePOWI {
    enum parser_exe_t type = PARSER_EXE_POWI;
    int i;
};

struct alignas(8) ParserExeFMA_VPV {
    enum parser_exe_t type = PARSER_EXE_FMA_VPV;
    int i;       // variable index
    double a;    // multiplier
    double b;    // addend
};

struct alignas(8) ParserExeIF {
    enum parser_exe_t type = PARSER_EXE_IF;
    int offset;
};

struct alignas(8) ParserExeJUMP {
    enum parser_exe_t type = PARSER_EXE_JUMP;
    int offset;
};

// Fused "push f(leaf)" instructions. P: variable, V: literal.

struct alignas(8) ParserExeF1_P {
    enum parser_exe_t type = PARSER_EXE_F1_P;
    parser_f1_t ftype;
    int i;
};

struct alignas(8) ParserExeSquare_P {
    enum parser_exe_t type = PARSER_EXE_SQUARE_P;
    int i;
};

struct alignas(8) ParserExePOWI_P {
    enum parser_exe_t type = PARSER_EXE_POWI_P;
    int i;
    int n;
};

struct alignas(8) ParserExeF2_PP {
    enum parser_exe_t type = PARSER_EXE_F2_PP;
    parser_f2_t ftype;
    int i1;
    int i2;
};

struct alignas(8) ParserExeF2_PV {
    enum parser_exe_t type = PARSER_EXE_F2_PV;
    parser_f2_t ftype;
    int i;
    double v;
};

struct alignas(8) ParserExeF2_VP {
    enum parser_exe_t type = PARSER_EXE_F2_VP;
    parser_f2_t ftype;
    int i;
    double v;
};

AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
double parser_math_powi (double d, int n)
{
    if (n == 0) { return 1.0; }
    if (n < 0) {
        d = 1.0/d;
        n = -n;
    }
    // Straight-line code for small exponents, with the same order of
    // multiplications as the loop below.
    switch (n) {
    case 1: return d;
    case 2: return d*d;
    case 3: return (d*d)*d;
    case 4: d *= d; return d*d;
    case 5: { double y = d; d *= d; d *= d; return d*y; }
    case 6: { d *= d; double y = d; d *= d; return d*y; }
    case 7: { double y = d; d *= d; y *= d; d *= d; return d*y; }
    case 8: d *= d; d *= d; return d*d;
    default: break;
    }
    double y = 1.0;
    while (n > 1) {
        if (n % 2 == 0) {
            d *= d;
            n = n/2;
        } else {
            y *= d;
            d *= d;
            n = (n-1)/2;
        }
    }
    return d * y;
}

AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
double parser_exe_eval (const char* p, double const* x)
{
    if (p == nullptr) { return std::numeric_limits<double>::max(); }
#define AMREXPR_PARSER_THREADED_BODY 0
#include "amrexpr_Parser_Exe_Body.H"
#undef AMREXPR_PARSER_THREADED_BODY
#undef AMREXPR_PARSER_CASE
#undef AMREXPR_PARSER_NEXT
}

#if AMREXPR_PARSER_THREADED_DISPATCH
// Host-only version with threaded dispatch, defined in
// amrexpr_Parser_Exe.cpp.  ParserExecutor uses it only for expressions
// long enough to amortize the call.
double parser_exe_eval_threaded (const char* p, double const* x);
#endif

void parser_compile_exe_size (struct parser_node* node, char*& p, std::size_t& exe_size,
                              int& max_stack_size, int& stack_size, std::vector<char const*>& local_variables);

inline std::size_t
parser_exe_size (struct amrexpr_parser* parser, int& max_stack_size, int& stack_size)
{
    parser_ast_sort(parser->ast);
    char* p = nullptr;
    std::size_t exe_size = 0;
    max_stack_size = 0;
    stack_size = 0;
    std::vector<char const*> local_variables;
    parser_compile_exe_size(parser->ast, p, exe_size, max_stack_size, stack_size, local_variables);
    stack_size -= static_cast<int>(local_variables.size())+1;
    return exe_size+sizeof(ParserExeNull);
}

inline std::vector<char const*>
parser_compile (struct amrexpr_parser* parser, char* p)
{
    std::size_t exe_size = 0;
    int max_stack_size = 0;
    int stack_size = 0;
    std::vector<char const*> local_variables;
    parser_compile_exe_size(parser->ast, p, exe_size, max_stack_size, stack_size, local_variables);
    new(p) ParserExeNull;
    return local_variables;
}

void parser_exe_print(char const* parser, std::vector<std::string> const& vars,
                      std::vector<char const*> const& locals);

}

#endif
