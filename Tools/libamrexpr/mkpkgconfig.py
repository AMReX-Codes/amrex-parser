#!/usr/bin/env python3

import argparse

def doit(prefix, version, cflags, libs, libpriv):
    print("# amrexpr Version: "+version)
    print("")
    print("prefix="+prefix)
    print("exec_prefix=${prefix}")
    print("libdir=${prefix}/lib")
    print("includedir=${prefix}/include")
    print("")
    print("Name: amrexpr")
    print("Description: AMReX's Math Expression Parser")
    print("Version:")
    print("URL: https://github.com/AMReX-Codes/amrex-parser")
    print("Requires:")
    print("Cflags: -I${includedir}", cflags)
    print("Libs: -L${libdir} -lamrexpr", libs)
    print("Libs.private:", libpriv)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        help="prefix",
                        default="")
    parser.add_argument("--version",
                        help="version",
                        default="")
    parser.add_argument("--cflags",
                        help="cflags",
                        default="")
    parser.add_argument("--libs",
                        help="libs",
                        default="")
    parser.add_argument("--libpriv",
                        help="libpriv",
                        default="")
    args = parser.parse_args()

    try:
        doit(prefix=args.prefix, version=args.version, cflags=args.cflags,
             libs=args.libs, libpriv=args.libpriv)
    except:
        # something went wrong
        print("$(error something went wrong in mkpkgconfig.py)")
