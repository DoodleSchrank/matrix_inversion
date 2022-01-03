#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex


def process(binary, algorithm, type, size, dbl = False):
    x = "_double" if dbl else ""
    with open("./output/" + binary[2:] + "_" + algorithm + "_" + type + "_" + str(size) + x, 'w') as f:
        for iter in range(5):
            out = Popen([binary, "./matrices/randomMatrix_" + type + "_" + str(size) + ".txt", str(size), algorithm, str(iter)], stdout=PIPE)
            f.writelines(str(out.stdout.read().decode('utf-8')))
    f.close()

gcc_location = "/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/gcc-11.2.0-sacutezrgifekoo3z5tmzaps6twen6jm/bin/g++"
clang_location = "/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/bin/clang++"
nvcc_location = "/usr/bin/nvc++"

types = ["normal", "natural", "sparse", "triangle", ]
algorithms_gcc = ["eigen", "openmp-cpu", "openmp-offload"]
algorithms_nvcc = ["openacc", "cuda", "cublas"]
sizes = [2**x for x in range(1,13)] #2^15 = 16384






# ensure single precision compilation
call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I /usr/include/eigen3/ -O2 -fopenmp -lOpenCL"))
call(shlex.split(clang_location + " -o clang-offload mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I /usr/include/eigen3/  -O2 -I /home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/include -lcudart -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib/libomptarget-nvptx-cuda_111-sm_61.bc -Wl,-rpath,/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib -B /usr/bin -lOpenCL"))
call(shlex.split(nvcc_location + " -o nvcc-offload mains/nvcc-offload.cu -Wall -Wextra -fast -acc -fopenmp -I /opt/nvidia/hpc_sdk/Linux_x86_64/21.5/math_libs/11.3/targets/x86_64-linux/include -lcublas"))

for binary in ["./gcc-offload", "./clang-offload", "./nvcc-offload"]:
    for algorithm in algorithms_gcc if binary != "./nvcc-offload" else algorithms_nvcc:
        for type in types:
            for size in sizes:
                process(binary, algorithm, type, size)


call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I /usr/include/eigen3/ -O2 -fopenacc -lOpenCL"))
binary = "./gcc-offload"
algorithm = "openacc"
for type in types:
    for size in sizes:
        process(binary, binary, type, size)





# double precision compilation
call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I /usr/include/eigen3/ -O2 -fopenmp -lOpenCL -DNAME=dbl"))
call(shlex.split(clang_location + " -o clang-offload mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I /usr/include/eigen3/  -O2 -I /home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/include -lcudart -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib/libomptarget-nvptx-cuda_111-sm_61.bc -Wl,-rpath,/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib -B /usr/bin -lOpenCL -DNAME=dbl"))
call(shlex.split(nvcc_location + " -o nvcc-offload mains/nvcc-offload.cpp -Wall -Wextra -fast -acc -fopenmp -I /opt/nvidia/hpc_sdk/Linux_x86_64/21.5/math_libs/11.3/targets/x86_64-linux/include -lcublas -DNAME=dbl"))


for binary in ["./gcc-offload", "./clang-offload", "./nvcc-offload"]:
    for algorithm in algorithms_gcc if binary != "./nvcc-offload" else algorithms_nvcc:
        for type in types:
            for size in sizes:
                process(binary, algorithm, type, size, True)

call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I /usr/include/eigen3/ -O2 -fopenacc -lOpenCL -DNAME=dbl"))
binary = "./gcc-offload"
algorithm = "openacc"
for type in types:
    for size in sizes:
        process(binary, binary, type, size, True)