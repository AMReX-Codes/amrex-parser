#ifndef AMREXPR_PARSER_H_
#define AMREXPR_PARSER_H_

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

template <int N>
struct ParserExecutor
{
    template <int M=N, std::enable_if_t<M==0,int> = 0>
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    double operator() () const noexcept
    {
        AMREXPR_IF_ON_DEVICE((return parser_exe_eval(m_device_executor, nullptr);))
        AMREXPR_IF_ON_HOST((return parser_exe_eval(m_host_executor, nullptr);))
    }

    template <typename... Ts>
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    std::enable_if_t<sizeof...(Ts) == N && !amrexpr::Same<float,Ts...>::value, double>
    operator() (Ts... var) const noexcept
    {
        double l_var[] = {var...};
        AMREXPR_IF_ON_DEVICE((return parser_exe_eval(m_device_executor, l_var);))
        AMREXPR_IF_ON_HOST((return parser_exe_eval(m_host_executor, l_var);))
    }

    template <typename... Ts>
    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    std::enable_if_t<sizeof...(Ts) == N &&  amrexpr::Same<float,Ts...>::value, float>
    operator() (Ts... var) const noexcept
    {
        double l_var[] = {var...};
        AMREXPR_IF_ON_DEVICE((return static_cast<float>(parser_exe_eval(m_device_executor, l_var));))
        AMREXPR_IF_ON_HOST((return static_cast<float>(parser_exe_eval(m_host_executor, l_var));))
    }

    [[nodiscard]] AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    double operator() (double var[N]) const noexcept
    {
        AMREXPR_IF_ON_DEVICE((return parser_exe_eval(m_device_executor, var);))
        AMREXPR_IF_ON_HOST((return parser_exe_eval(m_host_executor, var);))
    }

    AMREXPR_GPU_HOST_DEVICE AMREXPR_FORCE_INLINE
    explicit operator bool () const {
        AMREXPR_IF_ON_DEVICE((return m_device_executor != nullptr;))
        AMREXPR_IF_ON_HOST((return m_host_executor != nullptr;))
    }

    char* m_host_executor = nullptr;
#ifdef AMREXPR_USE_GPU
    char* m_device_executor = nullptr;
#endif
};

class Parser
{
public:
    Parser (std::string const& func_body);
    Parser () = default;
    void define (std::string const& func_body);

    explicit operator bool () const;

    void setConstant (std::string const& name, double c);

    void registerVariables (std::vector<std::string> const& vars);

    void print () const;
    void printExe () const;

    [[nodiscard]] int depth () const;
    [[nodiscard]] int maxStackSize () const;

    [[nodiscard]] std::string expr () const;

    [[nodiscard]] std::set<std::string> symbols () const;

    //! This compiles for both GPU and CPU
    template <int N> [[nodiscard]] ParserExecutor<N> compile () const;

    //! This compiles for CPU only
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
        assert(N == m_data->m_nvars);

        if (!(m_data->m_host_executor)) {
            int stack_size;
            m_data->m_exe_size = static_cast<int>
                (parser_exe_size(m_data->m_parser, m_data->m_max_stack_size,
                                 stack_size));

            if (m_data->m_max_stack_size > AMREXPR_PARSER_STACK_SIZE) {
                throw std::runtime_error("amrexpr::Parser: AMREXPR_PARSER_STACK_SIZE, "
                             + std::to_string(AMREXPR_PARSER_STACK_SIZE) + ", is too small for "
                             + m_data->m_expression);
            }
            if (stack_size != 0) {
                throw std::runtime_error("amrexpr::Parser: something went wrong with parser stack! "
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
