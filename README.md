# matrix_inversion
Currently compiles for g++:<br>
<code>g++ -o main main.cpp -I /usr/include/eigen3/ -O2 -fopenmp</code><br>
for nvc++:<br>
<code>nvc++ -o main main.cu -fast -acc -fopenmp</code><br>
in the respective folders.

Usable with <code>./matrix _size_ _alg_</code> where size determines the dimension of the matrix that is generated in the parent directory with the supplemented <code>random-matrix.py</code>.<br>
_alg_ in the g++ part is = 1 for Eigen, = 2 for CPU Single Core and = 4 for OpenMP. Adding Numbers together will execute multiple algorithms.<br>
_alg_ in the nvc++ part is = 1 for OpenACC and = 2 for CUDA.
