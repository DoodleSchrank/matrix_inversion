#!/bin/sh
#source /nfs/dust/atlas/user/maniageo/spack/share/spack/setup-env.sh

#spack load --first gcc@9.3.0
#spack load --first cuda@10.2.89 %gcc@9.3.0 arch=linux-centos7-skylake_avx512
#spack load llvm@12.0.0
#spack load cmake@3.18.0 %gcc@9.3.0
#spack load eigen@3.4.0
#spack load nvhpc@21.9

cd /nfs/dust/atlas/user/maniageo/ba/matrix_inversion/

run=('./gcc-offload-acc-dbl openacc'
'./gcc-offload-acc openacc'
'./gcc-offload-dbl openmp-offload'
'./gcc-offload openmp-offload'
'./nvcc-offload-dbl cuda'
'./nvcc-offload cuda'
'./nvcc-offload-dbl openacc'
'./nvcc-offload openacc'
'./nvcc-offload-dbl openmp-offload'
'./nvcc-offload openmp-offload'
'./clang-offload-dbl openmp-offload'
'./clang-offload openmp-offload'
'./gcc-offload-dbl eigen natural 2048'
'./gcc-offload-dbl eigen normal 8192'
'./gcc-offload-dbl eigen sparse 2048'
'./gcc-offload-dbl eigen triangle 2048'
'./gcc-offload-dbl openmp-cpu natural 2048'
'./gcc-offload-dbl openmp-cpu normal 2048'
'./gcc-offload-dbl openmp-cpu sparse 2048'
'./gcc-offload-dbl openmp-cpu triangle 2048'
'./gcc-offload-dbl openmp-offload natural 2048'
'./gcc-offload-dbl openmp-offload normal 2048'
'./gcc-offload-dbl openmp-offload sparse 2048'
'./gcc-offload-dbl openmp-offload triangle 2048'
'./gcc-offload eigen natural 4096'
'./gcc-offload eigen normal 8192'
'./gcc-offload eigen sparse 2048'
'./gcc-offload eigen triangle 2048'
'./gcc-offload openmp-cpu natural 2048'
'./gcc-offload openmp-cpu normal 2048'
'./gcc-offload openmp-cpu sparse 2048'
'./gcc-offload openmp-cpu triangle 2048'
'./nvcc-offload-dbl cublas natural 256'
'./nvcc-offload-dbl cublas normal 256'
'./nvcc-offload-dbl cublas sparse 256'
'./nvcc-offload-dbl cublas triangle 256'
'./nvcc-offload cublas natural 256'
'./nvcc-offload cublas normal 256'
'./nvcc-offload cublas sparse 256'
'./nvcc-offload cublas triangle 256'
'./clang-offload-dbl eigen natural 16384'
'./clang-offload-dbl eigen triangle 2048'
'./clang-offload-dbl openmp-cpu natural 1024'
'./clang-offload-dbl openmp-cpu normal 1024'
'./clang-offload-dbl openmp-cpu triangle 1024'
'./clang-offload-dbl openmp-cpu sparse 2048'
'./clang-offload eigen sparse 2048'
'./clang-offload eigen triangle 2048'
'./clang-offload openmp-cpu natural 1024'
'./clang-offload openmp-cpu sparse 2048'
'./clang-offload openmp-cpu triangle 1024')

python fix-benchmark-desy.py ${runs[$1]}
