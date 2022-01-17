#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "hip/hip_runtime.h"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

#define cudacall(call)                                                                                                        \
	do {                                                                                                                      \
		hipError_t err = (call);                                                                                             \
		if (hipSuccess != err) {                                                                                             \
			fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
			hipDeviceReset();                                                                                                \
			exit(EXIT_FAILURE);                                                                                               \
		}                                                                                                                     \
	} while (0)

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
		} else if (i < 2 * dim) {
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
	cudacall(hipMalloc(&d_A, dim * dim * sizeof(scalar)));
	cudacall(hipMalloc(&d_I, dim * dim * sizeof(scalar)));
	cudacall(hipMemcpy(d_A, A, dim * dim * sizeof(scalar), hipMemcpyHostToDevice));
	cudacall(hipMemcpy(d_I, I, dim * dim * sizeof(scalar), hipMemcpyHostToDevice));

	// setup kernel sizes
	struct hipDeviceProp_t properties;
	cudacall(hipGetDeviceProperties(&properties, 0));

	int threads = min(2 * dim, properties.maxThreadsPerBlock);
	dim3 norm_block(threads);
	dim3 norm_grid(1);

	threads = min(2 * dim, properties.maxThreadsPerBlock);
	dim3 gauss_block(threads);
	dim3 gauss_grid(1, dim);

	for (int iter = 0; iter < dim; iter++) {
		// swap lines if 0 -> divide by 0 is not allowed
		if (A[iter * dim + iter] == 0) {
			cudacall(hipLaunchKernelGGL(finddiagonal, norm_grid, norm_block, 0, 0, d_A, d_I, iter, dim));
		}

		//normalize
		cudacall(hipLaunchKernelGGL(normalize, norm_grid, norm_block, 0, 0, d_A, d_I, iter, dim));
		cudacall(hipDeviceSynchronize());

		//gauss
		cudacall(hipLaunchKernelGGL(gauss, gauss_grid, gauss_block, 0, 0, d_A, d_I, iter, dim));
		cudacall(hipDeviceSynchronize());
		cudacall(hipLaunchKernelGGL(gauss_fix, norm_grid, norm_block, 0, 0, d_A, iter, dim));
		cudacall(hipDeviceSynchronize());
	}

	// Copy results back to host
	cudacall(hipDeviceSynchronize());
	cudacall(hipMemcpy(I, d_I, dim * dim * sizeof(scalar), hipMemcpyDeviceToHost));
	cudacall(hipMemcpy(A, d_A, dim * dim * sizeof(scalar), hipMemcpyDeviceToHost));
	cudacall(hipFree(d_A));
	cudacall(hipFree(d_I));
}
