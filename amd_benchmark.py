#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex


clang_location = "/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/bin/clang++"

max_matrix = 8192
types = ["normal", "natural", "sparse", "triangle", ]
algorithms = ["eigen", "openmp-cpu", "openmp-offload", "hip"]
sizes = [2**x for x in range(1,13)] #2^15 = 16384

# ensure single precision compilation
call(shlex.split(clang_location + " -o clang-offload mains/amd.cpp -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -I /home/spackuser/yannik/eigen-3.4.0/Eigen  -O2 -I /home/spackuser/spack/opt/spack/linux-ubuntu18.04-piledriver/gcc-7.5.0/hip-4.3.1-ic4ohglkzzteoictju7bx3rhcgvqjtel/include  -Wl,-rpath,/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib -B /usr/bin"))

for binary in ["./clang-offload"]:
    for type in types:
        for algorithm in algorithms:
            for size in sizes:
                with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size), 'w') as f:
                    for iter in range(5):
                        out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
                        f.writelines(str(out.stdout.read().decode('utf-8')))
                    f.close()



# double precision compilation
call(shlex.split(clang_location + " -o clang-offload mains/amd.cpp -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -I /home/spackuser/yannik/eigen-3.4.0/Eigen  -O2 -I /home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/include -lcudart -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib/libomptarget-nvptx-cuda_111-sm_61.bc -Wl,-rpath,/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib -B /usr/bin -lOpenCL -DNAME=dbl"))

for binary in ["./clang-offload"]:
    for type in types:
        for algorithm in algorithms:
            for size in sizes:
                with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size), 'w') as f:
                    for iter in range(5):
                        out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + "_double.txt", str(size), algorithm, str(iter)], stdout=PIPE)
                        f.writelines(str(out.stdout.read().decode('utf-8')))
                    f.close()