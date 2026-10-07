#ifndef AMREXPR_PARSER_H_
#define AMREXPR_PARSER_H_

/**
 * \file amrexpr_Parser.H
 * \brief Runtime expression parser front-end and compiled executor helpers.
 */

#include "amrexpr_Arena.H"
#include "amrexpr_Gpu.H"
#include "amrexpr_Parser_Exe.H"
#include "amrexpr_REAL.H"
#include "amrexpr_TypeTraits.H"
#include "amrexpr_Utility.H"

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace amrexpr {

/**
 * \struct amrexpr::ParserExecutor
 * \brief Callable wrapper around a compiled parser expression with \p N variables.
 *
 * Instances are returned by Parser::compile() / Parser::compileHost() and may be
 * invoked on host or device. Device execution is available only when AMREXPR_USE_GPU
 * is enabled and a device executor buffer has been created.
 *
 * \tparam N Number of variables referenced in the expression.
 */
template <int N>
struct ParserExecutor
{
    /**
     * \brief Evaluate the expression with no arguments.
     *
     * Enabled only for constant expressions (\p N == 0).
     *
     * \return Expression value as double precision.
     */
    template <int M=N, std::enable_if_t<M==0,int> = 0>
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    double operator() () const noexcept
    {
        AMREXPR_IF_ON_DEVICE((return parser_exe_eval(m_device_executor, nullptr);))
        AMREXPR_IF_ON_HOST((return parser_exe_eval(m_host_executor, nullptr);))
    }

    /**
     * \brief Evaluate the expression with \p N double-compatible arguments.
     *
     * \param var Scalar arguments supplied in parser variable order.
     * \return Expression value promoted to double precision.
     */
    template <typename... Ts>
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    std::enable_if_t<sizeof...(Ts) == N && !amrexpr::Same<float,Ts...>::value, double>
    operator() (Ts... var) const noexcept
    {
        double l_var[] = {var...};
        AMREXPR_IF_ON_DEVICE((return parser_exe_eval(m_device_executor, l_var);))
        AMREXPR_IF_ON_HOST((return parser_exe_eval(m_host_executor, l_var);))
    }

    /**
     * \brief Evaluate the expression with \p N float arguments.
     *
     * Returns a float when all inputs are float.
     *
     * \param var Scalar arguments supplied in parser variable order.
     */
    template <typename... Ts>
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    std::enable_if_t<sizeof...(Ts) == N &&  amrexpr::Same<float,Ts...>::value, float>
    operator() (Ts... var) const noexcept
    {
        double l_var[] = {var...};
        AMREXPR_IF_ON_DEVICE((return static_cast<float>(parser_exe_eval(m_device_executor, l_var));))
        AMREXPR_IF_ON_HOST((return static_cast<float>(parser_exe_eval(m_host_executor, l_var));))
    }

    /**
     * \brief Evaluate the expression using an explicit array of inputs.
     * \param var Array storing the \p N variables in parser variable order.
     * \return Expression value promoted to double precision.
     */
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    double operator() (double var[N]) const noexcept
    {
        AMREXPR_IF_ON_DEVICE((return parser_exe_eval(m_device_executor, var);))
        AMREXPR_IF_ON_HOST((return parser_exe_eval(m_host_executor, var);))
    }

    /**
     * \brief True when a compiled executor buffer is available.
     */
    AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    explicit operator bool () const {
        AMREXPR_IF_ON_DEVICE((return m_device_executor != nullptr;))
        AMREXPR_IF_ON_HOST((return m_host_executor != nullptr;))
    }

    char* m_host_executor = nullptr;  ///< Pointer to host-visible bytecode.
#ifdef AMREXPR_USE_GPU
    char* m_device_executor = nullptr; ///< Pointer to device-visible bytecode (if copied).
#endif
};

/**
 * \brief Front-end for parsing scalar expressions into GPU/CPU executors.
 *
 * A Parser instance owns the expression string, parser state, and
 * registered variables. After calling define() and registerVariables(),
 * the expression can be compiled into a ParserExecutor that evaluates it
 * efficiently on host or device.
 */
class Parser
{
public:
    /**
     * \brief Construct a parser by immediately parsing \p func_body.
     * \param func_body Expression text; pass an empty string to defer parsing.
     */
    Parser (std::string const& func_body);
    //! Default-construct; call define() before compilation.
    Parser () = default;
    /**
     * \brief Parse and own a new expression, replacing any previously parsed one.
     * \param func_body Expression text to parse.
     */
    void define (std::string const& func_body);

    //! True when an expression has been parsed successfully.
    explicit operator bool () const;

    /**
     * \brief Bind a named constant to \p c.
     * \param name Identifier referenced inside the expression.
     * \param c    Constant value substituted into the expression.
     * \throws std::runtime_error if called after compile() or compileHost().
     */
    void setConstant (std::string const& name, double c);

    /**
     * \brief Register the ordered list of variable names used by the expression.
     *
     * The ordering determines how arguments are passed to ParserExecutor::operator().
     *
     * \param vars Variable names supplied in parser argument order.
     * \throws std::runtime_error if called after compile() or compileHost().
     */
    void registerVariables (std::vector<std::string> const& vars);

    //! Print a human-readable representation of the parser tree to stdout.
    void print () const;
    //! Dump the compiled bytecode to stdout for debugging.
    void printExe () const;

    /**
     * \brief Return the maximum tree depth of the parsed expression.
     * \return Depth measured in parse-tree levels (0 if undefined).
     */
    [[nodiscard]] int depth () const;
    /**
     * \brief Return the maximum stack usage recorded during compilation.
     * \return Stack depth in parser instructions (0 if undefined).
     */
    [[nodiscard]] int maxStackSize () const;

    /**
     * \brief Return the sanitized expression string currently managed by the Parser.
     * \return Copy of the parsed expression, or an empty string if undefined.
     */
    [[nodiscard]] std::string expr () const;

    /**
     * \brief Return the set of symbols (variables and constants) referenced by the expression.
     * \return Alphabetically sorted symbol names.
     */
    [[nodiscard]] std::set<std::string> symbols () const;

    /**
     * \brief Compile the current expression into a host/device executor.
     * \tparam N Number of registered variables.
     * \return ParserExecutor callable on both host and GPU (if available).
     */
    template <int N> [[nodiscard]] ParserExecutor<N> compile () const;

    /**
     * \brief Compile the current expression into a host-only executor.
     * \tparam N Number of registered variables.
     * \return ParserExecutor that executes only on the host.
     */
    template <int N> [[nodiscard]] ParserExecutor<N> compileHost () const;

private:

    struct Data {
        std::string m_expression;
        struct amrexpr_parser* m_parser = nullptr;
        int m_nvars = 0;
        mutable char* m_host_executor = nullptr;
#ifdef AMREXPR_USE_GPU
        mutable char* m_device_executor = nullptr;
#endif
        mutable int m_max_stack_size = 0;
        mutable int m_exe_size = 0;
        mutable std::vector<char const*> m_locals;
        Data () = default;
        ~Data ();
        Data (Data const&) = delete;
        Data (Data &&) = delete;
        Data& operator= (Data const&) = delete;
        Data& operator= (Data &&) = delete;
    };

    std::shared_ptr<Data> m_data;
    std::vector<std::string> m_vars;
};

template <int N>
ParserExecutor<N>
Parser::compileHost () const
{
    if (m_data && m_data->m_parser) {
        if (N != m_data->m_nvars) {
            throw std::runtime_error("amrexpr: the number of variables in compile() and registerVariables() do not match");
        }

        if (!(m_data->m_host_executor)) {
            int stack_size;
            m_data->m_exe_size = static_cast<int>
                (parser_exe_size(m_data->m_parser, m_data->m_max_stack_size,
                                 stack_size));

            if (m_data->m_max_stack_size > AMREXPR_PARSER_STACK_SIZE) {
                throw std::runtime_error("amrexpr: AMREXPR_PARSER_STACK_SIZE, "
                             + std::to_string(AMREXPR_PARSER_STACK_SIZE) + ", is too small for "
                             + m_data->m_expression);
            }
            if (stack_size != 0) {
                throw std::runtime_error("amrexpr: something went wrong with parser stack! "
                             + std::to_string(stack_size));
            }

            m_data->m_host_executor = (char*)amrexpr::allocate_host(m_data->m_exe_size);

            try {
                m_data->m_locals = parser_compile(m_data->m_parser,
                                                  m_data->m_host_executor);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(std::string(e.what()) + " in Parser expression \""
                                         + m_data->m_expression + "\"");
            }
        }

#ifdef AMREXPR_USE_GPU
        return ParserExecutor<N>{m_data->m_host_executor, m_data->m_device_executor};
#else
        return ParserExecutor<N>{m_data->m_host_executor};
#endif
    } else {
        return ParserExecutor<N>{};
    }
}

template <int N>
ParserExecutor<N>
Parser::compile () const
{
    auto exe = compileHost<N>();

#ifdef AMREXPR_USE_GPU
    if (m_data && m_data->m_parser && !(m_data->m_device_executor)) {
        m_data->m_device_executor = (char*)amrexpr::allocate_device(m_data->m_exe_size);
        Gpu::htod_memcpy(m_data->m_device_executor, m_data->m_host_executor, m_data->m_exe_size);
        exe.m_device_executor = m_data->m_device_executor;
    }
#endif

    return exe;
}

}

#endif
