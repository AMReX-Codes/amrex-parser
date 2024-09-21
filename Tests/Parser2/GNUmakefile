AMREXPR_HOME := ../..

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

include $(AMREXPR_HOME)/Tools/GNUMake/Make.defs

#ifeq ($(COMP),intel)
#  CXXFLAGS += -fp-model=precise
#endif

include ./Make.package
include $(AMREXPR_HOME)/Src/Make.package
include $(AMREXPR_HOME)/Tools/GNUMake/Make.rules
