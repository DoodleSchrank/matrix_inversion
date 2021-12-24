#!/usr/bin/python
from subprocess import Popen, PIPE

max_matrix = 8192
types = ["normal", "natural", "sparse", "triangle", ]
algorithms_gcc = ["eigen", "openmp-cpu", "openmp-offload", "openacc"]
algorithms_nvcc = ["openacc", "cuda", "cublas"]
sizes = [2**x for x in range(1,14)] #2^15 = 16384


for binary in ["./gcc-offload", "./clang-offload", "./nvcc-offload"]:
    for type in types:
        for algorithm in algorithms_gcc if binary != "./nvcc-offload" else algorithms_nvcc:
            for size in sizes:
                with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size), 'a') as f:
                    for iter in range(5):
                        out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
                        f.writelines(str(out.stdout.read().decode('utf-8')))
                    f.close()
