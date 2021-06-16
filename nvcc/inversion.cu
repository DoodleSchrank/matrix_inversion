#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "cuda_runtime.h"

#define cudaCheckErrors() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

template<int dim>
void openacc_offload(float matrix[dim][dim], float iden[dim][dim]) {
#pragma acc enter data copyin(matrix[0:dim][0:dim], iden[0:dim][0:dim])
	for (int i = 0; i < dim; i++) {
		if (matrix[i][i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j][i] != 0) {
#pragma acc parallel loop worker vector//vector_length(32)
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i][x] += matrix[j][x];
						iden[i][x] += iden[j][x];
					}
					break;
				}
			}
		}
		
		
		//normalize
#pragma acc parallel loop
		for (int x = i; x < dim + i + 1; x++) {
			float factor = matrix[i][i];
			if (x < dim)
				matrix[i][x] /= factor;
			else
				iden[i][x - dim] /= factor;
		}
		
		//gauss
#pragma acc parallel loop gang worker //vector_length(32)
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y][i];
			if (y != i && factor != 0.0f) {
#pragma acc loop vector
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y][x] -= matrix[i][x] * factor;
					else
						iden[y][x - dim] -= iden[i][x - dim] * factor;
				}
			}
		}
	}
#pragma acc exit data copyout(iden[0:dim][0:dim])
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
}


template<int dim>
void cuda_offload(float matrix[dim][dim], float iden[dim][dim]) {
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
	printf("#: %d, %d\n", blocks, dim);
	int block_size = floor(dim / blocks);
	printf("size: %d, %d, %d\n", block_size, dim, blocks);
	dim3 norm_block(block_size);
	dim3 norm_grid(1 + ceil(static_cast<double>(dim) / block_size));
	dim3 gauss_block(block_size);
	dim3 gauss_grid(2 * blocks, (int) ceil(static_cast<double>(dim) / gauss_block.y));
	/*int block_size = min(1024, dim);
	dim3 norm_block(block_size);
	dim3 norm_grid(1 +ceil((dim) / block_size));
	dim3 gauss_block(block_size);
	dim3 gauss_grid(2 * (int) ceil(dim / gauss_block.x), (int) ceil(dim / gauss_block.y));*/
	printf("%d - %d:%d\n", norm_block.x, norm_grid.x, norm_grid.y);
	printf("%d - %d:%d\n", gauss_block.x, gauss_grid.x, gauss_grid.y);
	
	for (int iter = 0; iter < dim; iter++) {
		if (matrix[iter][iter] == 0) { // swap lines if 0
			for (int j = iter + 1; j < dim; j++) { // find new line
				if (matrix[j][iter] != 0) {
					for (int x = iter; x < dim; x++) { // swap lines
						matrix[iter][x] += matrix[j][x];
						iden[iter][x] += iden[j][x];
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
