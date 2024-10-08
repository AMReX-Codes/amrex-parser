GNU_DOT_MAK_INCLUDED = TRUE

########################################################################

ifndef AMREXPR_CCOMP
  AMREXPR_CCOMP = gnu
endif

########################################################################

ifeq ($(USE_CUDA),TRUE)
  ifdef NVCC_CCBIN
    GCC_VERSION_COMP = $(NVCC_CCBIN)
  else
    GCC_VERSION_COMP = g++
  endif
else
  GCC_VERSION_COMP = $(CXX)
endif

gcc_version       = $(shell $(GCC_VERSION_COMP) -dumpfullversion -dumpversion | head -1 | sed -e 's;.*  *;;')
gcc_major_version = $(shell $(GCC_VERSION_COMP) -dumpfullversion -dumpversion | head -1 | sed -e 's;.*  *;;' | sed -e 's;\..*;;')
gcc_minor_version = $(shell $(GCC_VERSION_COMP) -dumpfullversion -dumpversion | head -1 | sed -e 's;.*  *;;' | sed -e 's;[^.]*\.;;' | sed -e 's;\..*;;')

COMP_VERSION = $(gcc_version)

########################################################################

GENERIC_GNU_FLAGS =

gcc_major_ge_8 = $(shell expr $(gcc_major_version) \>= 8)
gcc_major_ge_9 = $(shell expr $(gcc_major_version) \>= 9)
gcc_major_ge_10 = $(shell expr $(gcc_major_version) \>= 10)
gcc_major_ge_11 = $(shell expr $(gcc_major_version) \>= 11)
gcc_major_ge_12 = $(shell expr $(gcc_major_version) \>= 12)

INLINE_LIMIT ?= 43210

ifneq ($(NO_CONFIG_CHECKING),TRUE)
ifneq ($(gcc_major_ge_8),1)
  $(error GCC < 8 not supported)
endif
endif

ifeq ($(THREAD_SANITIZER),TRUE)
  GENERIC_GNU_FLAGS += -fsanitize=thread
endif
ifeq ($(FSANITIZER),TRUE)
  GENERIC_GNU_FLAGS += -fsanitize=address -fsanitize=undefined
  GENERIC_GNU_FLAGS += -fsanitize=pointer-compare -fsanitize=pointer-subtract
  GENERIC_GNU_FLAGS += -fsanitize=builtin -fsanitize=pointer-overflow
endif

########################################################################
########################################################################
########################################################################

ifeq ($(AMREXPR_CCOMP),gnu)

CXX = g++

CXXFLAGS =

########################################################################

CXXFLAGS += -Werror=return-type

ifeq ($(DEBUG),TRUE)
  CXXFLAGS += -g -O0 -ggdb -ftrapv
else
  CXXFLAGS += -g1 -O3
  ifneq ($(USE_COMPILER_DEFAULT_INLINE),TRUE)
    CXXFLAGS += -finline-limit=$(INLINE_LIMIT)
  endif
endif

ifeq ($(WARN_ALL),TRUE)
  warning_flags = -Wall -Wextra -Wlogical-op -Wfloat-conversion -Wnull-dereference -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wmissing-include-dirs

  ifeq ($(WARN_SIGN_COMPARE),FALSE)
    warning_flags += -Wno-sign-compare
  endif

  ifneq ($(USE_CUDA),TRUE)
    # With -Wpedantic I got 650 MB of warnings
    warning_flags += -Wpedantic
  endif

  ifneq ($(WARN_SHADOW),FALSE)
    warning_flags += -Wshadow
  endif

  ifeq ($(gcc_major_ge10),1)
    warning_flags += -Wextra-semi
  endif

  CXXFLAGS += $(warning_flags) -Woverloaded-virtual -Wnon-virtual-dtor
endif

ifeq ($(WARN_ERROR),TRUE)
  CXXFLAGS += -Werror
endif

ifeq ($(USE_GPROF),TRUE)
  CXXFLAGS += -pg
endif


ifeq ($(USE_COMPILE_PIC),TRUE)
  CXXFLAGS = -fPIC
endif

ifeq ($(ERROR_DEPRECATED),TRUE)
  CXXFLAGS += -Werror=deprecated
endif

########################################################################

ifdef CXXSTD
  CXXSTD := $(strip $(CXXSTD))
  CXXFLAGS += -std=$(CXXSTD)
else
  CXXFLAGS += -std=c++17
endif

########################################################################

CXXFLAGS += $(GENERIC_GNU_FLAGS) -pthread

endif # AMREXPR_CCOMP == gnu
