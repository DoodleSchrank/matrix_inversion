#include "cublas_v2.h"
#include "cuda_runtime.h"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

#define cudaCheckErrors()                                                                    \
	{                                                                                        \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e != cudaSuccess) {                                                              \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0);                                                                         \
		}                                                                                    \
	}


__global__ void finddiagonal(scalar *A, scalar *I, int iter, int dim) {
	int column = threadIdx.x;
	__shared__ int newline = 0;
	if (column == 0) {
		for (int row = iter + 1; row < dim; row++) {// find new line
			if (A[row * dim + iter] != 0) {
				newline = row;
			}
		}
	}
	__syncthreads();

	for (int i = column; i < 2 * dim; i += blockDim.x) {
		if (i < dim) {
			A[iter * dim + i] += A[newline * dim + i];
		} else {
			I[iter * dim + i - dim] += I[newline * dim + i - dim];
		}
	}
}

__global__ void normalize(scalar *A, scalar *I, int iter, int dim) {
	__shared__ scalar diag_elem = A[iter * dim + iter];
	__syncthreads();

	int column = threadIdx.x;

	for (int i = column; i < dim + iter + 1; i += blockDim.x) {
		if (i < dim) {
			A[iter * dim + i] /= diag_elem;
		} else if (i < 2 * dim) {
			I[iter * dim + i - dim] /= diag_elem;
		}
	}
}


__global__ void gauss(scalar *A, scalar *I, int iter, int dim) {
	int column = 1 + iter + blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (column >= 2 * dim || row == iter)
		return;

	scalar factor = A[row * dim + iter];

	for (int i = column; i < dim + iter + 1; i += blockDim.x) {
		if (i < dim) {
			A[row * dim + i] -= A[iter * dim + i] * factor;
		} else  if (i < 2 * dim) {
			I[row * dim + i - dim] -= I[iter * dim + i - dim] * factor;
		}
	}
}

__global__ void gauss_fix(scalar *A, int iter, int dim) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= dim || row == iter)
		return;
	for (int i = row; i < dim; i += blockDim.x) {
		A[i * dim + iter] = 0;
	}
}


void cuda_offload(scalar *A, scalar *I, int dim) {
	scalar *d_A, *d_I;

	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(scalar));
	cudaMalloc(&d_I, dim * dim * sizeof(scalar));
	cudaMemcpy(d_A, A, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, I, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice);
	cudaCheckErrors();

	// setup kernel sizes

	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);

	int threads = min(2 * dim, properties.maxThreadsPerBlock);
	dim3 norm_block(threads);
	dim3 norm_grid(1);
	
	threads = min(2 * dim, properties.maxThreadsPerBlock);
	dim3 gauss_block(threads);
	dim3 gauss_grid(1, dim);

	for (int iter = 0; iter < dim; iter++) {
		// swap lines if 0 -> divide by 0 is not allowed
		if (A[iter * dim + iter] == 0) {
			finddiagonal<<<norm_grid, norm_block>>>(d_A, d_I, iter, dim);
		}

		//normalize
		normalize<<<norm_grid, norm_block>>>(d_A, d_I, iter, dim);
		cudaDeviceSynchronize();
		cudaCheckErrors();

		//gauss
		gauss<<<gauss_grid, gauss_block>>>(d_A, d_I, iter, dim);
		cudaDeviceSynchronize();
		gauss_fix<<<norm_grid, norm_block>>>(d_A, iter, dim);
		cudaDeviceSynchronize();
		cudaCheckErrors();
	}
	cudaCheckErrors();

	// Copy results back to host
	cudaDeviceSynchronize();
	cudaCheckErrors();
	cudaMemcpy(I, d_I, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaMemcpy(A, d_A, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_I);
	cudaCheckErrors();
}
