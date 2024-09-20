AMREX_HOME := ../..

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(COMP),intel)
  CXXFLAGS += -fp-model=precise # otherwise Test 10 fails
endif

include ./Make.package
include $(AMREX_HOME)/Src/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
