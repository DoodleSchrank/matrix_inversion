#!/bin/sh
#source /nfs/dust/atlas/user/maniageo/spack/share/spack/setup-env.sh

#spack load --first gcc@9.3.0
#spack load --first cuda@10.2.89 %gcc@9.3.0 arch=linux-centos7-skylake_avx512
#spack load llvm@12.0.0
#spack load cmake@3.18.0 %gcc@9.3.0
#spack load eigen@3.4.0
#spack load nvhpc@21.9

cd /nfs/dust/atlas/user/maniageo/ba/matrix_inversion/

runs=('./gcc-offload eigen normal 8192' 
'./gcc-offload eigen natural' 
'./gcc-offload eigen triangle' 
'./gcc-offload eigen sparse' 
'./gcc-offload openmp-cpu' 
'./gcc-offload openmp-offload'
'./gcc-offload-dbl eigen normal 8192' 
'./gcc-offload-dbl eigen natural'
'./gcc-offload-dbl eigen triangle' 
'./gcc-offload-dbl eigen sparse' 
'./gcc-offload-dbl openmp-cpu'
'./gcc-offload-dbl openmp-offload'
'./clang-offload-dbl eigen sparse'
'./clang-offload-dbl eigen triangle'
'./clang-offload-dbl openmp-cpu'
'./clang-offload-dbl openmp-offload'
'./clang-offload eigen sparse 2048'
'./clang-offload eigen triangle'
'./clang-offload openmp-cpu'
'./clang-offload openmp-offload'
'./nvcc-offload openmp-offload'
'./nvcc-offload-dbl openmp-offload'
'./nvcc-offload cublas'
'./nvcc-offload-dbl cublas')

python fix-benchmark-desy.py ${runs[$1]}
