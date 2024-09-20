#
# Generic setup for using clang
#
CXX = clang++

CXXFLAGS =

########################################################################

clang_version       = $(shell $(CXX) --version | head -1 | sed -e 's/.*version.*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/')
clang_major_version = $(shell $(CXX) --version | head -1 | sed -e 's/.*version.*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/' | sed -e 's;\..*;;')
clang_minor_version = $(shell $(CXX) --version | head -1 | sed -e 's/.*version.*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/' | sed -e 's;[^.]*\.;;' | sed -e 's;\..*;;')

COMP_VERSION = $(clang_version)

########################################################################

ifeq ($(DEBUG),TRUE)
  CXXFLAGS += -g -O0 -ftrapv
else
  CXXFLAGS += -g1 -O3
endif

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

# disable some warnings
CXXFLAGS += -Wno-c++17-extensions

########################################################################

ifdef CXXSTD
  CXXSTD := $(strip $(CXXSTD))
else
  CXXSTD := c++17
endif

CXXFLAGS += -std=$(CXXSTD)

########################################################################

GENERIC_COMP_FLAGS =

ifeq ($(EXPORT_DYNAMIC),TRUE)
  CPPFLAGS += -DAMREX_EXPORT_DYNAMIC
  LIBRARIES += -Xlinker -export_dynamic
  GENERIC_COMP_FLAGS += -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer
endif

ifeq ($(THREAD_SANITIZER),TRUE)
  GENERIC_COMP_FLAGS += -fsanitize=thread
endif
ifeq ($(FSANITIZER),TRUE)
  GENERIC_COMP_FLAGS += -fsanitize=address -fsanitize=undefined
endif

CXXFLAGS += -pthread

########################################################################

ifeq ($(FSANITIZER),TRUE)
  ifneq ($(shell uname),Darwin)
    override XTRALIBS += -lubsan
  endif
endif
