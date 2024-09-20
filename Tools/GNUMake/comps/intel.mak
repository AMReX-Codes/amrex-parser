#
# Generic setup for using Intel LLVM compiler
#
CXX = icpx

CXXFLAGS =

########################################################################

intel_version = $(shell $(CXX) -dumpversion)

COMP_VERSION = $(intel_version)

########################################################################

ifeq ($(DEBUG),TRUE)
  CXXFLAGS += -g -O0 -ftrapv
else
  CXXFLAGS += -g1 -O3
endif

########################################################################

ifeq ($(WARN_ALL),TRUE)
  warning_flags = -Wall -Wextra -Wno-sign-compare -Wunreachable-code -Wnull-dereference
  warning_flags += -Wfloat-conversion -Wextra-semi

  ifneq ($(USE_CUDA),TRUE)
    warning_flags += -Wpedantic
  endif

  ifneq ($(WARN_SHADOW),FALSE)
    warning_flags += -Wshadow
  endif

  CXXFLAGS += $(warning_flags) -Woverloaded-virtual -Wnon-virtual-dtor
endif

ifeq ($(WARN_ERROR),TRUE)
  CXXFLAGS += -Werror
endif

CXXFLAGS += -Wno-tautological-constant-compare

########################################################################

ifdef CXXSTD
  CXXSTD := $(strip $(CXXSTD))
  CXXFLAGS += -std=$(CXXSTD)
else
  CXXFLAGS += -std=c++17
endif

########################################################################

CXXFLAGS += -pthread



