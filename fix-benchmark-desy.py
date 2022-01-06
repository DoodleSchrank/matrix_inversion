#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex
import sys

def process(binary, algorithm, type, size, dbl = False):
    #print("{} processing {} with matrix type {} and size {}. Its {}\n", binary, algorithm, type, size, "double" if dbl else "single")
    x = "_double" if dbl else ""
    with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size) + x, 'w') as f:
        for iter in range(5):
            out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
    f.close()

binariesl = ["./gcc-offload", "./clang-offload", "./nvcc-offload", "./gcc-offload-acc", "./gcc-offload-dbl", "./clang-offload-dbl", "./nvcc-offload-dbl", "./gcc-offload-acc-dbl"]
typesl = ["normal", "natural", "sparse", "triangle"]
algorithms_gccl = ["eigen", "openmp-cpu", "openmp-offload"]
algorithms_nvccl = ["openacc", "cuda", "cublas"]
sizesl = [2**x for x in range(1,15)] #2^15 = 16384

binary = sys.argv[1]
if len(sys.argv > 1):
    algorithms = [sys.argv[2]]
else:
    algorithms = algorithms_nvccl if binary = binariesl[2] or binary = binariesl[6] else ["openacc"] if binary = binariesl[3] or binary = binariesl[7] else algorithms_gccl

if len(sys.argv > 2):
    types = [sys.argv[3]]
else:
    types = typesl

if len(sys.argv > 3):
    sizes = sizesl[math.log(int(sys.argv[4]), 2):]
else:
    sizes = sizesl

dbl = True if binary[-3:] == "dbl" else False

for algorithm in algorithms:
    for type in types:
        for size in sizes:
            process(binary, algorithm, type, size, dbl)