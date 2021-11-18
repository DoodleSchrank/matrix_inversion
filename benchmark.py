#!/usr/bin/python
from subprocess import Popen, PIPE

max_matrix = 16384


algo = "eigen"
size = 2
while size <= max_matrix:
    with open("./output/" + algo + "_" + str(size), 'a') as f:
        for iter in range(10):
            out = Popen(["./gcc/main", str(size), str(1), str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
        size = size * 2
        f.close()

algo = "cpu"
size = 2
while size <= max_matrix:
    with open("./output/" + algo + "_" + str(size), 'a') as f:
        for iter in range(10):
            out = Popen(["./gcc/main", str(size), str(2), str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
        size = size * 2
        f.close()

algo = "openmp"
size = 2
while size <= max_matrix:
    with open("./output/" + algo + "_" + str(size), 'a') as f:
        for iter in range(10):
            out = Popen(["./gcc/main", str(size), str(4), str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
        size = size * 2
        f.close()

algo = "openacc"
size = 2
while size <= max_matrix:
    with open("./output/" + algo + "_" + str(size), 'a') as f:
        for iter in range(10):
            out = Popen(["./nvcc/main", str(size), str(1), str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
        size = size * 2
        f.close()

algo = "cuda"
size = 2
while size <= max_matrix:
    with open("./output/" + algo + "_" + str(size), 'a') as f:
        for iter in range(10):
            out = Popen(["./nvcc/main", str(size), str(2), str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
        size = size * 2
        f.close()
