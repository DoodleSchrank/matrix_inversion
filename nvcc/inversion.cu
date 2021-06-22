#include <stdio.h>
#include "cuda_runtime.h"

#define cudaCheckErrors() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

void openacc_offload(float* matrix, float* iden, int dim) {
	float factor;
	int x, y, i, j;
	int size = dim * dim;
#pragma acc data copyin(matrix[0:size], iden[0:size]) copyout(iden[0:size]) create(factor, x, y, i, j)
	for (i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] != 0) {
#pragma acc parallel loop worker vector//vector_length(32)
					for (x = i; x < dim; x++) { // swap lines
						matrix[i * dim + x] += matrix[j * dim + x];
						iden[i * dim + x] += iden[j * dim + x];
					}
					break;
				}
			}
		}
		
		//normalize
#pragma acc serial
{
		factor = matrix[i * dim + i];
	};
#pragma acc parallel loop
		for (x = i; x < dim + i + 1; x++) {
			factor = matrix[i * dim + i];
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else {
				iden[i * dim + x - dim] /= factor;
			}
		}
		
		//gauss
#pragma acc parallel loop gang worker //vector_length(32)
		for (y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
#pragma acc loop vector
				for (x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y * dim + x] -= matrix[i * dim + x] * factor;
					else
						iden[y * dim + x - dim] -= iden[i * dim + x - dim] * factor;
				}
			}
		}
	}
//#pragma acc exit data copyout(iden[0:dim*dim])
}


__global__ void normalize(float *matrix, float *iden, int iter, int dim) {
	int x = iter + blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= 2 * dim)
		return;
	
	float factor = matrix[iter * dim + iter];
	if (x < dim)
		matrix[iter * dim + x] /= factor;
	else if (x < 2 * dim) {
		iden[iter * dim + x - dim] /= factor;
	}
	__syncthreads();
}


__global__ void gauss(float *matrix, float *iden, int iter, int dim) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float factor = matrix[y * dim + iter];
	if (x >= 2 * dim || y >= dim || factor == 0.0f || y == iter)
		return;
	
	if (x < dim) {
		matrix[y * dim + x] -= matrix[iter * dim + x] * factor;
	} else {
		iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;
	}
	__syncthreads();
}


void cuda_offload(float* matrix, float* iden, int dim) {
	auto h_A = reinterpret_cast<float *>(matrix);
	auto h_I = reinterpret_cast<float *>(iden);
	float *d_A, *d_I;
	
	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_I, dim * dim * sizeof(float));
	cudaCheckErrors();
	cudaMemcpy(d_A, h_A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors();
	
	// setup kernel kernel
	int blocks = ceil(static_cast<double>(dim) / 1024.0f);
	//printf("#: %d, %d\n", blocks, dim);
	int block_size = floor(dim / blocks);
	//printf("size: %d, %d, %d\n", block_size, dim, blocks);
	dim3 norm_block(block_size);
	dim3 norm_grid(1 + ceil(static_cast<double>(dim) / block_size));
	dim3 gauss_block(block_size);
	dim3 gauss_grid(2 * blocks, (int) ceil(static_cast<double>(dim) / gauss_block.y));
	/*int block_size = min(1024, dim);
	dim3 norm_block(block_size);
	dim3 norm_grid(1 +ceil((dim) / block_size));
	dim3 gauss_block(block_size);
	dim3 gauss_grid(2 * (int) ceil(dim / gauss_block.x), (int) ceil(dim / gauss_block.y));*/
	//printf("%d - %d:%d\n", norm_block.x, norm_grid.x, norm_grid.y);
	//printf("%d - %d:%d\n", gauss_block.x, gauss_grid.x, gauss_grid.y);
	
	for (int iter = 0; iter < dim; iter++) {
		if (matrix[iter * dim + iter] == 0) { // swap lines if 0
			for (int j = iter + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + iter] != 0) {
					for (int x = iter; x < dim; x++) { // swap lines
						matrix[iter * dim + x] += matrix[j * dim + x];
						iden[iter * dim + x] += iden[j * dim + x];
					}
					break;
				}
			}
		}
		
		//normalize
		normalize<<<norm_grid, norm_block>>>(d_A, d_I, iter, dim);
		cudaCheckErrors();
		
		//gauss
		gauss<<<gauss_grid, gauss_block>>>(d_A, d_I, iter, dim);
		cudaCheckErrors();
	}
	cudaCheckErrors();
	
	// Copy results back to host
	cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
}
