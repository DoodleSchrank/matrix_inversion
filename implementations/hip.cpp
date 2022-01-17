#include "hip/hip_runtime.h"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif


__global__ void finddiagonal(scalar *A, scalar *I, int iter, int dim) {
	int x = threadIdx.x;
	__shared__ int newline;
	newline = 0;
	if (x == 0) {
		for (int j = iter + 1; j < dim; j++) {// find new line
			if (A[j * dim + iter] != 0) {
				newline = j;
			}
		}
	}
	__syncthreads();

	for (int i = x; i < 2 * dim; i += blockDim.x) {
		if (i < dim) {
			scalar temp = A[iter * dim + i];
			A[iter * dim + i] = A[newline * dim + i];
			A[newline * dim + i] = temp;
		} else {
			scalar temp = I[iter * dim + i - dim];
			I[iter * dim + i - dim] = I[newline * dim + i - dim];
			I[newline * dim + i - dim] = temp;
		}
	}
}

__global__ void normalize(scalar *A, scalar *I, int iter, int dim) {
	__shared__ scalar diag_elem;
	diag_elem = A[iter * dim + iter];
	__syncthreads();

	int x = threadIdx.x;

	for (int i = x; i < 2 * dim; i += blockDim.x) {
		if (i < dim)
			A[iter * dim + i] /= diag_elem;
		else if (i < 2 * dim)
			I[iter * dim + i - dim] /= diag_elem;
	}
}


__global__ void gauss(scalar *A, scalar *I, int iter, int dim) {
	int x = 1 + iter + blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= 2 * dim || y == iter)
		return;

	scalar factor = A[y * dim + iter];


	if (x < dim)
		A[y * dim + x] -= A[iter * dim + x] * factor;
	else
		I[y * dim + x - dim] -= I[iter * dim + x - dim] * factor;
}

__global__ void gauss_fix(scalar *A, int iter, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= dim || x == iter)
		return;

	A[x * dim + iter] = 0;
}


void hip_offload(scalar *A, scalar *I, int dim) {
	scalar *d_A, *d_I;
	printf("1: starting offload; next: memcpy\n");
	// setup and copy matrices to gpu
	hipMalloc(&d_A, dim * dim * sizeof(scalar));
	hipMalloc(&d_I, dim * dim * sizeof(scalar));
	hipMemcpy(d_A, A, dim * dim * sizeof(scalar), hipMemcpyHostToDevice);
	hipMemcpy(d_I, I, dim * dim * sizeof(scalar), hipMemcpyHostToDevice);

	printf("2: successfull memcpy; next: kernel parameters\n");
	// setup kernelsizes

	struct hipDeviceProp_t properties;
	hipGetDeviceProperties(&properties, 0);

	int row_parts = 1;
	int threads = min(2 * dim, properties.maxThreadsPerBlock);
	dim3 norm_block(threads);
	dim3 norm_grid(row_parts);
	row_parts = (2 * dim > properties.maxThreadsPerBlock) ? std::ceil(2. * dim / properties.maxThreadsPerBlock) : 1;
	threads = std::ceil(2. * dim / row_parts);
	dim3 gauss_block(threads);
	dim3 gauss_grid(row_parts, dim);
	
	printf("3: successfull kernel parameters; next: algorithm\n");
	for (int iter = 0; iter < dim; iter++) {
		if (A[iter * dim + iter] == 0) {// swap lines if 0 -> divide by 0 is impossible
			hipLaunchKernelGGL(finddiagonal, norm_grid, norm_block, sizeof(int), 0, A, I, iter, dim);
		}
		
		printf("a: iter: %d after finddiagonal\n");

		//normalize
		hipLaunchKernelGGL(normalize, norm_grid, norm_block, sizeof(scalar), 0, d_A, d_I, iter, dim);
		hipDeviceSynchronize();
		printf("a: after normalize\n");

		//gauss
		hipLaunchKernelGGL(gauss, gauss_grid, gauss_block, 0, 0, d_A, d_I, iter, dim);
		hipDeviceSynchronize();
		printf("a: after gauss\n");
		hipLaunchKernelGGL(gauss_fix, norm_grid, norm_block, 0, 0, d_A, iter, dim);
		hipDeviceSynchronize();
		printf("a: after gaussfix\n");
	}

	// Copy results back to host
	hipDeviceSynchronize();
	hipMemcpy(I, d_I, dim * dim * sizeof(scalar), hipMemcpyDeviceToHost);
	hipMemcpy(A, d_A, dim * dim * sizeof(scalar), hipMemcpyDeviceToHost);
	hipFree(d_A);
	hipFree(d_I);
	printf("4: after memcpy and free\n");
}
