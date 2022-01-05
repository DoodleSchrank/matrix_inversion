#include "hip/hip_runtime.h"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif


__global__ void finddiagonal(scalar *matrix, scalar *iden, int iter, int dim) {
	int x = threadIdx.x;
	__shared__ int newline = 0;
	if (x == 0) {
		for (int j = iter + 1; j < dim; j++) {// find new line
			if (matrix[j * dim + iter] != 0) {
				newline = j;
			}
		}
	}
	__syncthreads();

	for (int i = x; i < 2 * dim; i += blockDim.x) {
		if (i < dim) {
			scalar temp = matrix[iter * dim + i];
			matrix[iter * dim + i] = matrix[newline * dim + i];
			matrix[newline * dim + i] = temp;
		} else {
			scalar temp = iden[iter * dim + i - dim];
			iden[iter * dim + i - dim] = iden[newline * dim + i - dim];
			iden[newline * dim + i - dim] = temp;
		}
	}
}

__global__ void normalize(scalar *matrix, scalar *iden, int iter, int dim) {
	__shared__ scalar diag_elem = matrix[iter * dim + iter];
	__syncthreads();

	int x = threadIdx.x;

	for (int i = x; i < 2 * dim; i += blockDim.x) {
		if (i < dim)
			matrix[iter * dim + i] /= diag_elem;
		else if (i < 2 * dim)
			iden[iter * dim + i - dim] /= diag_elem;
	}
}


__global__ void gauss(scalar *matrix, scalar *iden, int iter, int dim) {
	int x = 1 + iter + blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= 2 * dim || y == iter)
		return;

	scalar factor = matrix[y * dim + iter];


	if (x < dim)
		matrix[y * dim + x] -= matrix[iter * dim + x] * factor;
	else
		iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;
}

__global__ void gauss_fix(scalar *matrix, int iter, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= dim || x == iter)
		return;

	matrix[x * dim + iter] = 0;
}


void hip_offload(scalar *matrix, scalar *iden, int dim) {
	scalar *d_A, *d_I;

	// setup and copy matrices to gpu
	hipMalloc(&d_A, dim * dim * sizeof(scalar));
	hipMalloc(&d_I, dim * dim * sizeof(scalar));
	hipMemcpy(d_A, matrix, dim * dim * sizeof(scalar), hipMemcpyHostToDevice);
	hipMemcpy(d_I, iden, dim * dim * sizeof(scalar), hipMemcpyHostToDevice);

	// setup kernelsizes

	struct hipDeviceProp properties;
	hipGetDeviceProperties(&properties, 0);

	int row_parts = 1;
	int threads = min(2 * dim, properties.maxThreadsPerBlock);
	dim3 norm_block(threads);
	dim3 norm_grid(row_parts);
	row_parts = (2 * dim > properties.maxThreadsPerBlock) ? std::ceil(2. * dim / properties.maxThreadsPerBlock) : 1;
	threads = std::ceil(2. * dim / row_parts);
	dim3 gauss_block(threads);
	dim3 gauss_grid(row_parts, dim);

	for (int iter = 0; iter < dim; iter++) {
		if (matrix[iter * dim + iter] == 0) {// swap lines if 0 -> divide by 0 is impossible
			hipLaunchKernel(finddiagonal, norm_grid, norm_block, matrix, iden, iter, dim);
		}

		//normalize
		hipLaunchKernel(normalize, norm_grid, norm_block, d_A, d_I, iter, dim);
		hipDeviceSynchronize();

		//gauss
		hipLaunchKernel(gauss, gauss_grid, gauss_block, d_A, d_I, iter, dim);
		hipDeviceSynchronize();
		hipLaunchKernel(gauss_fix, norm_grid, norm_block, d_A, iter, dim);
		hipDeviceSynchronize();
	}

	// Copy results back to host
	hipDeviceSynchronize();
	hipMemcpy(iden, d_I, dim * dim * sizeof(scalar), hipMemcpyDeviceToHost);
	hipMemcpy(matrix, d_A, dim * dim * sizeof(scalar), hipMemcpyDeviceToHost);
	hipFree(d_A);
	hipFree(d_I);
}
