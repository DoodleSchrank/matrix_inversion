# matrix_inversion algorithm
Implemented with Eigen, cuBLAS, OpenMP, OpenACC, CUDA as well as HIP and OpenCL.

Currently compiles for g++:
<br>
<code>g++ -o gcc-offload mains/gcc-offload.cpp -I /usr/include/eigen3/ -O2 -fopenmp</code><br>
for nvc++:
<br>
<code>nvc++ -o nvcc-offload mains/nvcc-offload.cu -acc -fopenmp -I /opt/nvidia/hpc_sdk/Linux_x86_64/21.5/math_libs/11.3/targets/x86_64-linux/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/math_libs/11.3/targets/x86_64-linux/lib64 -lcublas</code>
<br>
and for clang++ with
<br>
<code>clang++ -o clang-offload mains/clang-offload.cpp -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I /usr/include/eigen3/ -O2 -I <spack_location>/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/include -lcudart -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/lib64 -ldl -lrt -pthreads --libomptarget-nvptx-bc-path=<spack_location>/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib/libomptarget-nvptx-cuda_111-sm_61.bc -Wl,-rpath,<spack_location>/linux-archrolling-skylake/gcc-11.1.0/llvm-12.0.1-ovixtokbztjgh2ggohidxigccgpb6wxc/lib -B /usr/bin -lOpenCL </code>
<br>
in the respective folders.
  
The thesis.pdf gives more detailed description on what this repository is about.
