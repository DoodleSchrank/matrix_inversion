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

gcc_location = "/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/gcc-11.1.0-qyyagvftz5pgrbuxefahh273x5txjpml/bin/g++"
clang_location = "/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/bin/clang++"
nvcc_location = "/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/compilers/bin/nvc++"

binaries = ["./gcc-offload", "./clang-offload", "./nvcc-offload"]
types = ["normal", "natural", "sparse", "triangle"]
algorithms_gcc = ["eigen", "openmp-cpu", "openmp-offload", "openacc"]
algorithms_nvcc = ["openacc", "cuda", "cublas"]
sizes = [2**x for x in range(1,15)] #2^15 = 16384

run = int(sys.argv[1])
dbl = 0 if run < 44 else 1
type = types[run % 4]
binary = binaries[0] if run % 11 < 4 else binaries[1] if run % 11 < 8 else binaries[2]
algorithm = algorithms_gcc[(run % 11) % 4] if run % 11 < 8 else algorithms_nvcc[(run % 11) % 4]

if algorithm == "openacc":
    if binary == "./clang-offload":
        sys.exit(0)
    if dbl == 0:
        call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -fopenacc"))
    else:
        call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -fopenacc -DNAME=dbl"))

if run == 0:
    # single precision compilation
    call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I  -O2 -fopenmp"))
    call(shlex.split(clang_location + " -o clang-offload mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/include/ -lcudart -L/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib/libomptarget-nvptx-cuda_102-sm_70.bc -Wl,-rpath,/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib -B /usr/bin"))
    call(shlex.split(nvcc_location + " -o nvcc-offload mains/nvcc-offload.cu -Wall -Wextra -fast -acc -fopenmp -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/math_libs/10.2/targets/x86_64-linux/include -lcublas"))
elif run == 44:
    # double precision compilation
    call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I  -O2 -fopenmp -DNAME=dbl"))
    call(shlex.split(clang_location + " -o clang-offload mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/include/ -lcudart -L/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib/libomptarget-nvptx-cuda_102-sm_70.bc -Wl,-rpath,/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib -B /usr/bin -DNAME=dbl"))
    call(shlex.split(nvcc_location + " -o nvcc-offload mains/nvcc-offload.cu -Wall -Wextra -fast -acc -fopenmp -I /nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/math_libs/10.2/targets/x86_64-linux/include -lcublas -DNAME=dbl"))

for size in sizes:
    process(binary, algorithm, type, size, dbl)
