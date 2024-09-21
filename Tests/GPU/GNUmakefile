AMREXPR_HOME := ../..

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_CUDA  = TRUE
USE_HIP   = FALSE
USE_SYCL  = FALSE

USE_CPU = FALSE
ifeq ($(USE_CPU),TRUE)
  USE_CUDA  = FALSE
  USE_HIP   = FALSE
  USE_SYCL  = FALSE
endif

include $(AMREXPR_HOME)/Tools/GNUMake/Make.defs
include ./Make.package
include $(AMREXPR_HOME)/Src/Make.package
include $(AMREXPR_HOME)/Tools/GNUMake/Make.rules
