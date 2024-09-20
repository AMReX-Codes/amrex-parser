#
# Generic setup for using dpcpp
#
CXX = icpx

amrex_oneapi_version = $(shell $(CXX) --version | head -1)
$(info oneAPI version: $(amrex_oneapi_version))

CXXFLAGS =

########################################################################

ifeq ($(DEBUG),TRUE)
  CXXFLAGS += -g -O0 #-ftrapv
else
  CXXFLAGS += -gline-tables-only -fdebug-info-for-profiling -O3
endif

ifeq ($(WARN_ALL),TRUE)
  warning_flags = -Wall -Wextra -Wno-sign-compare -Wunreachable-code -Wnull-dereference
  warning_flags += -Wfloat-conversion -Wextra-semi

  warning_flags += -Wpedantic

  # /tmp/icpx-2d34de0e47/global_vars-header-4390fb.h:25:36: error: zero size arrays are an extension [-Werror,-Wzero-length-array]
  #    25 | const char* const kernel_names[] = {
  #       |                                    ^
  # 1 error generated.
  #
  # Seen in oneapi 2024.2.0 after adding Test/DeviceGlobal
  warning_flags += -Wno-zero-length-array

  ifneq ($(WARN_SHADOW),FALSE)
    warning_flags += -Wshadow
  endif

  CXXFLAGS += $(warning_flags) -Woverloaded-virtual
endif

# disable warning: comparison with infinity always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
#                  return std::isinf(m);
# appeared since 2021.4.0
CXXFLAGS += -Wno-tautological-constant-compare

ifeq ($(WARN_ERROR),TRUE)
  CXXFLAGS += -Werror
endif

########################################################################

ifdef CXXSTD
  CXXFLAGS += -std=$(strip $(CXXSTD))
else
  CXXFLAGS += -std=c++17
endif

CXXFLAGS += -fsycl

ifneq ($(SYCL_SPLIT_KERNEL),FALSE)
  CXXFLAGS += -fsycl-device-code-split=per_kernel
endif

# temporary work-around for oneAPI beta08 bug
#   define "long double" as 64bit for C++ user-defined literals
#   https://github.com/intel/llvm/issues/2187
CXXFLAGS += -mlong-double-64 -Xclang -mlong-double-64

########################################################################

CXXFLAGS += -pthread

########################################################################

LDFLAGS += -fsycl-device-lib=libc,libm-fp32,libm-fp64

ifdef SYCL_PARALLEL_LINK_JOBS
LDFLAGS += -fsycl-max-parallel-link-jobs=$(SYCL_PARALLEL_LINK_JOBS)
endif

ifeq ($(SYCL_AOT),TRUE)
  ifndef AMREX_INTEL_ARCH
    ifdef INTEL_ARCH
      AMREX_INTEL_ARCH = $(INTEL_ARCH)
    endif
  endif
  ifdef AMREX_INTEL_ARCH
    amrex_intel_gpu_target = $(AMREX_INTEL_ARCH)
  else
    # amrex_intel_gpu_target = *
    $(error Either INTEL_ARCH or AMREX_INTEL_ARCH must be specified when SYCL_AOT is TRUE.)
  endif
  CXXFLAGS += -fsycl-targets=spir64_gen
  amrex_sycl_backend_flags = -device $(amrex_intel_gpu_target)
  SYCL_AOT_GRF_MODE ?= Default
  ifneq ($(SYCL_AOT_GRF_MODE),Default)
    ifeq ($(SYCL_AOT_GRF_MODE),Large)
      amrex_sycl_backend_flags += -internal_options -ze-opt-large-register-file
    else ifeq ($(SYCL_AOT_GRF_MODE),AutoLarge)
      amrex_sycl_backend_flags += -options -ze-intel-enable-auto-large-GRF-mode
    else
      $(error SYCL_AOT_GRF_MODE ($(SYCL_AOT_GRF_MODE)) must be either Default, Large, or AutoLarge)
    endif
  endif
  LDFLAGS += -Xsycl-target-backend '$(amrex_sycl_backend_flags)'
endif

ifeq ($(DEBUG),TRUE)
  # This might be needed for linking device code larger than 2GB.
  LDFLAGS += -fsycl-link-huge-device-code
endif

AMREX_CCACHE_ENV = CCACHE_DEPEND=1
