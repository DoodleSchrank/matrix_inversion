#include <stdio.h>
#include "cuda_runtime.h"

#define cudaCheckErrors() {                                          \
cudaError_t e=cudaGetLastError();                                 \
if(e!=cudaSuccess) {                                              \
printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
exit(0); \
}                                                                 \
}

void print_matrix(float *matrix, float *iden, int dim) {
	/*for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			printf("%2f ", matrix[i * dim + j]);
		}
		printf("\t\t");
		for (int j = 0; j < dim; j++) {
			printf("%2f ", iden[i * dim + j]);
		}
		printf("\n");
	}
	printf("\n");*/
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


__global__ void normalize(float *matrix, float *iden, int iter, int dim, float divisor) {
	int x = iter + blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= 2 * dim)
		return;
	
	//float factor = matrix[iter * dim + iter];
	if (x < dim)
		matrix[iter * dim + x] /= divisor;
	else if (x < 2 * dim) {
		iden[iter * dim + x - dim] /= divisor;
	}
	__syncthreads();
}


__global__ void gauss(float *matrix, float *iden, int iter, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
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


void cuda_offload(float *matrix, float *iden, int dim) {
	float *h_A = matrix;
	float *h_I = iden;
	float *d_A, *d_I;
	
	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_I, dim * dim * sizeof(float));
	cudaCheckErrors();
	cudaMemcpy(d_A, h_A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors();
	/*for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%2f ", h_A[i*dim+j]);
		}
		printf("\t\t");
		for (int j = 0; j < 4; j++) {
			printf("%2f ", h_I[i*dim+j]);
		}
		printf("\n");
	}
	printf("\n--------------------------------\n\n");*/
	
	// setup kernelsizes
	
	//TODO: ERROR (NAN) WHEN row_parts > 2!!
	//automatically happens when matrix dim is > 1024
	//can be forced on lower dims too.
	
	int row_parts = (dim + 1 > 1024) ? std::ceil((dim + 1.) / 1024.) : 1;
	int max_row = std::ceil((dim + 1.) / row_parts);
	dim3 norm_block(max_row);
	dim3 norm_grid(row_parts);
	row_parts = (2 * dim > 1024) ? std::ceil(2 * dim / 1024.) : 1;
	max_row = std::ceil(2. * dim / row_parts);
	dim3 gauss_block(max_row);
	dim3 gauss_grid(row_parts, dim);
	
	cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	
	
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
		
		//cudaMemcpy(&h_A[iter * dim + iter], &d_A[iter * dim + iter], sizeof(float), cudaMemcpyDeviceToHost);
		//normalize
		normalize<<<norm_grid, norm_block>>>(d_A, d_I, iter, dim, h_A[iter * dim + iter]);
		//cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		//print_matrix(h_A, h_I, dim);
		cudaCheckErrors();
		
		//gauss
		gauss<<<gauss_grid, gauss_block>>>(d_A, d_I, iter, dim);
		//cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		//print_matrix(h_A, h_I, dim);
		cudaCheckErrors();
	}
	cudaCheckErrors();
	
	// Copy results back to host
	cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	//print_matrix(h_A, h_I, dim);
}
