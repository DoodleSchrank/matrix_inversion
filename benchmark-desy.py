#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex
import sys
import math

def process(binary, algorithm, type, size, dbl = False):
		file = open("log.txt", "a")
		strin = binary + " processing " + algorithm + " with matrix type " + type + " and size " + str(size) + ". It's " + ("double." if dbl else "single.")
		print(strin)
		file.write(strin + "\n")
		file.close()
		x = "_double" if dbl else ""
		max = 5 if size < 1024 else 1
		with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size) + x, 'w') as f:
				for iter in range(max):
						iter = 4 if max == 1 else iter
						out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
						f.writelines(str(out.stdout.read().decode('utf-8')))
		f.close()

binaries = ["./gcc-offload", "./clang-offload", "./nvcc-offload", "./gcc-offload-acc", "./gcc-offload-dbl", "./clang-offload-dbl", "./nvcc-offload-dbl", "./gcc-offload-acc-dbl"]
types = ["normal", "natural", "sparse", "triangle"]
algorithms_gcc = ["eigen", "openmp-cpu", "openmp-offload"]
algorithms_nvcc = ["openacc", "cuda", "cublas", "openmp-offload"]
sizes = [2**x for x in range(1,14)] #2^15 = 16384

# get binary
binary = sys.argv[1]

# get algorithm, if given
if len(sys.argv) > 2 and sys.argv[2] != "all":
		algorithms = [sys.argv[2]]
else:
		algorithms = algorithms_nvcc if binary == binaries[2] or binary == binaries[6] else ["openacc"] if binary == binaries[3] or binary == binaries[7] else algorithms_gcc

# get types if given
if len(sys.argv) > 3 and sys.argv[3] != "all":
		types = [sys.argv[3]]

# get sizes if given
if len(sys.argv) > 4:
		sizes = sizes[int(math.log(int(sys.argv[4]), 2)) - 1:]

# get double if given
dbl = True if binary[-3:] == "dbl" else False

for algorithm in algorithms:
		for type in types:
				for size in sizes:
						process(binary, algorithm, type, size, dbl)
