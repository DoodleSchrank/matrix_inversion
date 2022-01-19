#!/bin/sh
#source /nfs/dust/atlas/user/maniageo/spack/share/spack/setup-env.sh

#spack load --first gcc@9.3.0
#spack load --first cuda@10.2.89 %gcc@9.3.0 arch=linux-centos7-skylake_avx512
#spack load llvm@12.0.0
#spack load cmake@3.18.0 %gcc@9.3.0
#spack load eigen@3.4.0
#spack load nvhpc@21.9

cd /nfs/dust/atlas/user/maniageo/ba/matrix_inversion/

run=(
'./clang-offload-dbl openmp-cpu all 1024'
'./clang-offload-dbl openmp-offload all 1024'
'./clang-offload all all 1024'
'./gcc-offload-dbl eigen normal 16384'
'./gcc-offload-dbl eigen natural 1024'
'./gcc-offload-dbl eigen sparse 1024'
'./gcc-offload-dbl eigen triangle 1024'
'./gcc-offload eigen normal 16384'
'./gcc-offload eigen natural 1024'
'./gcc-offload eigen sparse 1024'
'./gcc-offload eigen triangle 1024'
'./nvcc-offload-dbl'
'./nvcc-offload')

python fix-benchmark-desy.py ${runs[$1]}
