#include <stdio.h>
#include "cuda_runtime.h"

#define cudaCheckErrors() {                                          \
cudaError_t e=cudaGetLastError();                                 \
if(e!=cudaSuccess) {                                              \
printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
exit(0); \
}                                                                 \
}

__host__ __device__ void
print_matrix(float *matrix, float *iden, int dim, int xto, int yto, int xfrom = 0, int yfrom = 0) {
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

__host__ __device__ void print_matrix(float *matrix, float *iden, int dim, int xfrom = 0, int yfrom = 0) {
	print_matrix(matrix, iden, dim, dim, dim, xfrom, yfrom);
}


void openacc_offload(float *matrix, float *iden, int dim) {
#pragma acc data copy(matrix[0:dim * dim], iden[0:dim * dim])
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] != 0) {
#pragma acc parallel loop worker vector//vector_length(32)
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i * dim + x] += matrix[j * dim + x];
						iden[i * dim + x] += iden[j * dim + x];
					}
					break;
				}
			}
		}
		
		//normalize

#pragma acc parallel loop gang worker vector
		for (int x = i + 1; x < dim + i + 1; x++) {
			float factor = matrix[i * dim + i];
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else {
				iden[i * dim + x - dim] /= factor;
			}
		}
#pragma acc serial
		{
			matrix[i * dim + i] = 1;
		};
		
		//gauss
#pragma acc parallel loop gang worker device_type(nvidia)
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
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


__global__ void normalize(float *matrix, float *iden, int iter, int dim) {
	int x = 1 + iter + blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= 2 * dim)
		return;
	
	
	if (x < dim)
		matrix[iter * dim + x] /= matrix[iter * dim + iter];
	else
		iden[iter * dim + x - dim] /= matrix[iter * dim + iter];
}

__global__ void gauss(float *matrix, float *iden, int iter, int dim) {
	//int iterations_per_thread = std::ceil((dim + 1.) / 128.);
	int x = 1 + iter + blockIdx.x * blockDim.x + threadIdx.x;// * iterations_per_thread;
	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= 2 * dim || y == iter)
		return;
	
	//if (y >= iter) y++;
	float factor = matrix[y * dim + iter];
	
	//for (; x <= iter + (threadIdx.x + 1) * iterations_per_thread && x <= iter + dim; x++) {
		if (x < dim)
			matrix[y * dim + x] -= matrix[iter * dim + x] * factor;
		else
			iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;
	//}
	
	//matrix[y * dim + iter] = 0;
}

__global__ void gauss_fix(float *matrix, int iter, int dim) {
	//int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x >= dim || x == iter)
		return;
	
	//matrix[y * dim + iter] = 0;
	matrix[x * dim + iter] = 0;
}


void cuda_offload(float *matrix, float *iden, int dim) {
	float *d_A, *d_I;
	
	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_I, dim * dim * sizeof(float));
	cudaMemcpy(d_A, matrix, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, iden, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors();
	
	// setup kernelsizes
	
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	
	int row_parts = (dim + 1 > properties.maxThreadsPerBlock) ? std::ceil((dim + 1.) / properties.maxThreadsPerBlock) : 1;
	int max_row = std::ceil((dim + 1.) / row_parts);
	dim3 norm_block(max_row);
	dim3 norm_grid(row_parts);
	row_parts = (2 * dim > properties.maxThreadsPerBlock) ? std::ceil(2. * dim / properties.maxThreadsPerBlock) : 1;
	max_row = std::ceil(2. * dim / row_parts);
	dim3 gauss_block(max_row);
	dim3 gauss_grid(row_parts, dim);
	
	for (int iter = 0; iter < dim; iter++) {
		if (matrix[iter * dim + iter] == 0) { // swap lines if 0 -> divide by 0 is impossible
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
		cudaDeviceSynchronize();
		cudaCheckErrors();
		matrix[iter * dim + iter] = 1;
		cudaMemcpy(&d_A[iter * dim + iter], &matrix[iter * dim + iter], sizeof(float), cudaMemcpyHostToDevice);
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
	cudaMemcpy(iden, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrix, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_I);
	cudaCheckErrors();
}

