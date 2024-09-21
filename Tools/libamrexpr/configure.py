#!/usr/bin/env python3

import sys
import argparse

def configure(argv):
    argv[0] = "configure" # So the help message print it
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        help="Install libamrexpr and headers in PREFIX directory [default=tmp_install_dir]",
                        default="tmp_install_dir")
    parser.add_argument("--with-cuda",
                        help="Use CUDA [default=no]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--with-hip",
                        help="Use HIP [default=no]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--with-sycl",
                        help="Use SYCL [default=no]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--comp",
                        help="Compiler [default=gnu]",
                        choices=["gnu","intel","llvm"],
                        default="gnu")
    parser.add_argument("--debug",
                        help="Debug build [default=no]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--single-precision",
                        help="Define amrexpr::Real as float [default=no (i.e., double)]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--allow-different-compiler",
                        help="Allow an application to use a different compiler than the one used to build libamrexpr [default=no]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--enable-pic",
                        help="Enable position independent code [default=no]",
                        choices=["yes","no"],
                        default="no")
    parser.add_argument("--cuda-arch",
                        help="Specify CUDA architecture. Required when CUDA is enabled.")
    parser.add_argument("--amd-arch",
                        help="Specify AMD GPU architecture. Requried when HIP is enabled.")
    parser.add_argument("--intel-arch",
                        help="Specify Intel GPU architecture. Optional.")
    args = parser.parse_args()

    f = open("GNUmakefile","w")
    f.write("AMREXPR_INSTALL_DIR = " + args.prefix.strip() + "\n")
    f.write("USE_CUDA = {}\n".format("FALSE" if args.with_cuda == "no" else "TRUE"))
    f.write("USE_HIP = {}\n".format("FALSE" if args.with_hip == "no" else "TRUE"))
    f.write("USE_SYCL = {}\n".format("FALSE" if args.with_sycl == "no" else "TRUE"))
    f.write("COMP = " + args.comp.strip() + "\n")
    f.write("DEBUG = {}\n".format("TRUE" if args.debug == "yes" else "FALSE"))
    f.write("PRECISION = {}\n".format("FLOAT" if args.single_precision == "yes" else "DOUBLE"))
    f.write("ALLOW_DIFFERENT_COMP = {}\n".format("FALSE" if args.allow_different_compiler == "no" else "TRUE"))
    f.write("USE_COMPILE_PIC = {}\n".format("FALSE" if args.enable_pic == "no" else "TRUE"))
    if args.with_cuda == "yes":
        f.write("CUDA_ARCH = " + args.cuda_arch.strip() + "\n")
    if args.with_hip == "yes":
        f.write("AMD_ARCH = " + args.amd_arch.strip() + "\n")
    if args.with_sycl == "yes":
        if args.intel_arch:
            f.write("INTEL_ARCH = " + args.intel_arch.strip() + "\n")

    f.write("\n")

    fin = open("GNUmakefile.in","r")
    for line in fin.readlines():
        f.write(line)
    fin.close()

    f.close()

if __name__ == "__main__":
    configure(sys.argv)
