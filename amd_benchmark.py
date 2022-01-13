#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex
import sys
import math

def process(binary, algorithm, type, size, dbl = False):
    print(binary + " processing " + algorithm + " with matrix type " + type + " and size " + str(size) + ". It's " + ("double." if dbl else "single."))
    x = "_double" if dbl else ""
    #max = 5 if size < 1024 and algorithm not in ["openacc", "openmp-offload", "cuda", "cublas"] else 1
    max = 5 if size < 1024 else 1
    with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size) + x, 'w') as f:
        for iter in range(max):
            out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
    f.close()

binaries = ["./aomp-offload", "./aomp-offload-dbl", "./hip-offload", "./hip-offload", "./hip-offload-dbl"]
types = ["normal", "natural", "sparse", "triangle"]
sizes = [2**x for x in range(1,15)]

# get binary
binary = sys.argv[1]

# get algorithm, if given
algorithm = binary[2:6] if binary[2] == 'a' else binary[2:5]

# get types if given
if len(sys.argv) > 2:
    types = [sys.argv[2]]

# get sizes if given
if len(sys.argv) > 3:
    sizes = sizes[math.log(int(sys.argv[3]), 2):]

# get double if given
dbl = True if binary[-3:] == "dbl" else False

for type in types:
    for size in sizes:
        process(binary, algorithm, type, size, dbl)
