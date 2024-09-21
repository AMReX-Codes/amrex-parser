#!/usr/bin/env python3

import sys, re
import argparse

def doit(defines, undefines, comp, allow_diff_comp):
    print("#ifndef AMREXPR_HAVE_NO_CONFIG_H")
    print("#define AMREXPR_HAVE_NO_CONFIG_H")

    # Remove -I from input
    defines = re.sub(r'-I.*?(-D|$)', r'\1', defines)

    defs = defines.split("-D")
    for d in defs:
        dd = d.strip()
        if dd:
            v = dd.split("=")
            print("#ifndef",v[0])
            if len(v) == 2:
                print("#define",v[0],v[1])
            else:
                print("#define",v[0],1)
            print("#endif")

    for ud in undefines:
        print("#undef",ud)

    print("#ifdef __cplusplus");

    if allow_diff_comp == "FALSE":
        if comp == "gnu":
            comp_macro = "__GNUC__"
            comp_id    = "GNU"
        elif comp == "intel":
            comp_macro = "__INTEL_COMPILER"
            comp_id    = "Intel"
        elif comp == "llvm":
            comp_macro = "__llvm__"
            comp_id    = "Clang/LLVM"
        elif comp == "hip":
            comp_macro = "__HIP__"
            comp_id    = "HIP"
        elif comp == "sycl":
            comp_macro = "__INTEL_CLANG_COMPILER"
            comp_id    = "SYCL"
        else:
            sys.exit("ERROR: unknown compiler "+comp+" to mkconfig.py")

        msg = "#error libamrexpr was built with " + comp_id + ". "
        msg = msg + "To avoid this error, reconfigure with --allow-different-compiler=yes"
        print("#ifndef " + comp_macro )
        print(msg)
        print("#endif")

    print("#endif") #  ifdef __cplusplus

    print("#endif")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--defines",
                        help="preprocessing macros: -Dxx -Dyy",
                        default="")
    parser.add_argument("--undefines",
                        help="preprocessing macros to be undefined",
                        default="")
    parser.add_argument("--comp",
                        help="compiler",
                        choices=["gnu","intel","llvm","hip","sycl"])
    parser.add_argument("--allow-different-compiler",
                        help="allow an application to use a different compiler than the one used to build libamrexpr",
                        choices=["TRUE","FALSE"])
    args = parser.parse_args()

    try:
        doit(defines=args.defines, undefines=args.undefines, comp=args.comp,
             allow_diff_comp=args.allow_different_compiler)
    except:
        # something went wrong
        print("$(error something went wrong in mkconfig.py)")
