#include <stdio.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#define cudaCheckErrors() {                                          \
cudaError_t e=cudaGetLastError();                                 \
if(e!=cudaSuccess) {                                              \
printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
exit(0); \
}                                                                 \
}

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

__host__ __device__ void
print_matrix(scalar *matrix, scalar *iden, int dim, int xto, int yto, int xfrom = 0, int yfrom = 0) {
	for (int i = yfrom; i < yto; i++) {
		for (int j = xfrom; j < xto; j++) {
			printf("%2f ", matrix[i * dim + j]);
		}
		printf("\t\t");
		for (int j = xfrom; j < xto; j++) {
			printf("%2f ", iden[i * dim + j]);
		}
		printf("\n");
	}
	printf("\n");
}

__host__ __device__ void print_matrix(scalar *matrix, scalar *iden, int dim, int xfrom = 0, int yfrom = 0) {
	print_matrix(matrix, iden, dim, dim, dim, xfrom, yfrom);
}


void openacc_offload(scalar *matrix, scalar *iden, int dim) {
#pragma acc data copy(matrix[0:dim * dim], iden[0:dim * dim])
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] != 0) {
#pragma acc parallel loop worker vector
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i * dim + x] += matrix[j * dim + x];
						iden[i * dim + x] += iden[j * dim + x];
					}
					break;
				}
			}
		}
		
		//normalize
		scalar factor;
#pragma acc parallel loop gang worker vector device_type(nvidia)
		for (int x = i; x < dim + i + 1; x++) {
			factor = matrix[i * dim + i];
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else {
				iden[i * dim + x - dim] /= factor;
			}
		}
		
		//gauss
#pragma acc parallel loop gang worker device_type(nvidia)
		for (int y = 0; y < dim; y++) {
			scalar factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
#pragma acc loop vector
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y * dim + x] -= matrix[i * dim + x] * factor;
					else
						iden[y * dim + x - dim] -= iden[i * dim + x - dim] * factor;
				}
			}
		}
	}
}




void cublas_offload(scalar *matrix, scalar *iden, int dim) {
	scalar *d_A, *d_I;
	
	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(scalar));
	cudaMalloc(&d_I, dim * dim * sizeof(scalar));
	cudaMemcpy(d_A, matrix, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, iden, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice);
	cudaCheckErrors();
	
	// setup kernelsizes
	int info[1];
	cublasHandle_t cu_handle;
	cublasCreate(&cu_handle);
	scalar * const* matrices = &matrix;
	scalar * const* identities = &iden;
	
	cublasSgetrfBatched(cu_handle, dim, matrices, dim, NULL, info, 1);
	cublasSgetriBatched(cu_handle, dim, matrices, dim, NULL, identities, dim, info, 1);
	cudaCheckErrors();
	cudaMemcpy(iden, d_I, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrix, d_A, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_I);
	cudaCheckErrors();
}

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
		else
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


void cuda_offload(scalar *matrix, scalar *iden, int dim) {
	scalar *d_A, *d_I;

	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(scalar));
	cudaMalloc(&d_I, dim * dim * sizeof(scalar));
	cudaMemcpy(d_A, matrix, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, iden, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice);
	cudaCheckErrors();

	// setup kernelsizes

	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);

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
			finddiagonal<<<norm_grid, norm_block>>>(matrix, iden, iter, dim);
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
	cudaMemcpy(iden, d_I, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrix, d_A, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_I);
	cudaCheckErrors();
}
