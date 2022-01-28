#!/usr/bin/python
from subprocess import Popen, PIPE, call
import shlex

spack_location = "/nfs/dust/atlas/user/maniageo/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.3.0/"
gcc_location = spack_location + "gcc-11.1.0-qyyagvftz5pgrbuxefahh273x5txjpml/bin/g++"
clang_location = spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/bin/clang++"
nvcc_location = spack_location + "nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/compilers/bin/nvc++"

call(shlex.split(gcc_location + " -o gcc-offload-acc mains/gcc-offload.cpp -I " + spack_location + "eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -fopenacc"))
call(shlex.split(gcc_location + " -o gcc-offload-acc-dbl mains/gcc-offload.cpp -I " + spack_location + "eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -fopenacc -D dbl"))
call(shlex.split(gcc_location + " -o gcc-offload mains/gcc-offload.cpp -I  -O2 -fopenmp"))
call(shlex.split(clang_location + " -o clang-offload mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I " + spack_location + "eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -I " + spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/include/ -lcudart -L" + spack_location + "nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=" + spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib/libomptarget-nvptx-cuda_102-sm_70.bc -Wl,-rpath," + spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib -B /usr/bin"))
call(shlex.split(nvcc_location + " -o nvcc-offload mains/nvcc-offload.cu -Wall -Wextra -fast -acc -fopenmp -I " + spack_location + "nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/math_libs/10.2/targets/x86_64-linux/include -lcublas -gpu=cc70"))
call(shlex.split(gcc_location + " -o gcc-offload-dbl mains/gcc-offload.cpp -I  -O2 -fopenmp -D dbl"))
call(shlex.split(clang_location + " -o clang-offload-dbl mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I " + spack_location + "eigen-3.4.0-57ptt6xndy7fmu3f6w6uzuvgl74z56tx/include/eigen3  -O2 -I " + spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/include/ -lcudart -L" + spack_location + "nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=" + spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib/libomptarget-nvptx-cuda_102-sm_70.bc -Wl,-rpath," + spack_location + "llvm-12.0.0-fmx2razucqcja44det235qdbqelf2soi/lib -B /usr/bin -D dbl"))
call(shlex.split(nvcc_location + " -o nvcc-offload-dbl mains/nvcc-offload.cu -Wall -Wextra -fast -acc -fopenmp -I " + spack_location + "nvhpc-21.9-yiw3cngl4t4nwambluvyxdq2ctmgcmch/Linux_x86_64/21.9/math_libs/10.2/targets/x86_64-linux/include -lcublas -D dbl -gpu=cc70"))