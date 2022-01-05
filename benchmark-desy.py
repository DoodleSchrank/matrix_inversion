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

binaries = ["./gcc-offload", "./clang-offload", "./nvcc-offload", "./gcc-offload-acc", "./gcc-offload-dbl", "./clang-offload-dbl", "./nvcc-offload-dbl", "./gcc-offload-acc-dbl"]
types = ["normal", "natural", "sparse", "triangle"]
algorithms_gcc = ["eigen", "openmp-cpu", "openmp-offload"]
algorithms_nvcc = ["openacc", "cuda", "cublas"]
sizes = [2**x for x in range(1,15)] #2^15 = 16384

run = int(sys.argv[1])
binary = binaries[run]
dbl = True if run > 3 else False
algorithms = algorithms_nvcc if run == 2 or run == 6 else ["openacc"] if run == 3 or run == 7 else algorithms_gcc

for algorithm in algorithms:
    for type in types:
        for size in sizes:
            process(binary, algorithm, type, size, dbl)
