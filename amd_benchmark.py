#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex
import sys

def process(binary, algorithm, type, size, dbl = False):
    x = "_double" if dbl else ""
    with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size) + x, 'w') as f:
        for iter in range(5):
            out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
    f.close()

clang_location = "/home/spackuser/usr/lib/aomp/bin/clang++"
hip_location = ""

binaries = ["./hip-offload", "./clang-offload"]
types = ["normal", "natural", "sparse", "triangle"]
algorithms_gcc = ["openmp-offload", "hip"]
sizes = [2**x for x in range(1,15)] #2^15 = 16384

call(shlex.split(clang_location + " -o clang-offload mains/amd-offload.cpp -fopenmp"))
call(shlex.split(hip_location + " -o hip-offload mains/hip-offload.cpp -fopenmp"))
for type in types:
    for size in sizes:
        process("./hip-offload", "hip", type, size)
for type in types:
    for size in sizes:
        process("./clang-offload", "openmp-offload", type, size)

call(shlex.split(clang_location + " -o clang-offload mains/amd-offload.cpp -fopenmp -DNAME=dbl"))
call(shlex.split(hip_location + " -o hip-offload mains/hip-offload.cpp -fopenmp -DNAME=dbl"))
for type in types:
    for size in sizes:
        process("./hip-offload", "hip", type, size, True)
for type in types:
    for size in sizes:
        process("./clang-offload", "openmp-offload", type, size, True)
