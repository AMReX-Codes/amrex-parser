# Setup for HIP, using hipcc (HCC and clang will use the same compiler name).

ifneq ($(NO_CONFIG_CHECKING),TRUE)
  HIP_PATH=$(realpath $(shell hipconfig --path))
  hipcc_version := $(shell hipcc --version | grep "HIP version: " | cut -d" " -f3)
  hipcc_major_version := $(shell hipcc --version | grep "HIP version: " | cut -d" " -f3 | cut -d. -f1)
  hipcc_minor_version := $(shell hipcc --version | grep "HIP version: " | cut -d" " -f3 | cut -d. -f2)
  ifeq ($(HIP_PATH),)
    $(error hipconfig failed. Is the HIP toolkit available?)
  endif
  COMP_VERSION = $(hipcc_version)
endif

CXX = $(HIP_PATH)/bin/hipcc

ifdef CXXSTD
  CXXSTD := $(strip $(CXXSTD))
else
  CXXSTD := c++17
endif

# Generic flags, always used
CXXFLAGS = -std=$(CXXSTD) -m64

# rdc support
ifeq ($(USE_GPU_RDC),TRUE)
  HIPCC_FLAGS += -fgpu-rdc
endif

# amd gpu target
HIPCC_FLAGS += --offload-arch=$(AMD_ARCH)

# pthread
HIPCC_FLAGS += -pthread

CXXFLAGS += $(HIPCC_FLAGS)

# =============================================================================================

ifeq ($(DEBUG),TRUE)
  CXXFLAGS += -g -O1
else
  CXXFLAGS += -gline-tables-only -fdebug-info-for-profiling -O3
endif

ifeq ($(WARN_ALL),TRUE)
  warning_flags = -Wall -Wextra -Wunreachable-code -Wnull-dereference
  warning_flags += -Wfloat-conversion -Wextra-semi

  warning_flags += -Wpedantic

  ifneq ($(WARN_SHADOW),FALSE)
    warning_flags += -Wshadow
  endif

  CXXFLAGS += $(warning_flags) -Woverloaded-virtual
  CFLAGS += $(warning_flags)
endif

ifeq ($(WARN_ERROR),TRUE)
  CXXFLAGS += -Werror -Wno-deprecated-declarations -Wno-gnu-zero-variadic-macro-arguments
  CFLAGS += -Werror
endif

# Generic HIP info
ROC_PATH=$(realpath $(dir $(HIP_PATH)))
SYSTEM_INCLUDE_LOCATIONS += $(ROC_PATH)/include $(HIP_PATH)/include

# hipcc passes a lot of unused arguments to clang
LEGACY_DEPFLAGS += -Wno-unused-command-line-argument

