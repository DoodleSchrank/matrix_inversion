#include <stdio.h>
#include "cuda_runtime.h"

#define cudaCheckErrors() {                                          \
cudaError_t e=cudaGetLastError();                                 \
if(e!=cudaSuccess) {                                              \
printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
exit(0); \
}                                                                 \
}

__host__ __device__ void print_matrix(float *matrix, float *iden, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			printf("%2f ", matrix[i * dim + j]);
		}
		printf("\t\t");
		for (int j = 0; j < dim; j++) {
			printf("%2f ", iden[i * dim + j]);
		}
		printf("\n");
	}
	printf("\n");
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
	else {
		iden[iter * dim + x - dim] /= divisor;
	}
	if (x == 7) {
		//printf("%04f / %04f = %04f\n", iden[iter * dim + x - dim] * divisor, divisor, iden[iter * dim + x - dim]);
		//print_matrix(matrix, iden, dim);
	}
}


__global__ void gauss(float *matrix, float *iden, int iter, int dim) {
	int x = iter + blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	y = (y >= iter) ? y + 1 : y;
	
	float factor = matrix[y * dim + iter];
	if (x >= 2 * dim || y >= dim || factor == 0.0f || x == iter)
		return;
	
	if (x < dim) {
		matrix[y * dim + x] -= matrix[iter * dim + x] * factor;
	} else {
		iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;
	}
	
	//cudaDeviceSynchronize();
	
	//if (x == 0 && y == 0) {
		//print_matrix(matrix, iden, dim);
		//printf("%04f - %04f * %04f -> ", iden[y * dim + x - dim], iden[iter * dim + x - dim], factor);
	//}
}

__global__ void gauss_fix(float *matrix, int iter, int dim) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (y >= dim || y == iter)
		return;
	
	matrix[y * dim + iter] = 0;
}


/*__device__ cuda_kernel(float *matrix, float *iden, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
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
		float divisor = matrix[iter * dim + iter];
		
		if (y == 0) {
			if (x < dim)
				matrix[iter * dim + x] /= divisor;
			else if (x < 2 * dim) {
				iden[iter * dim + x - dim] /= divisor;
			}
		}
		
		normalize<<<norm_grid, norm_block>>>(d_A, d_I, iter, dim, h_A[iter * dim + iter]);
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		print_matrix(h_A, h_I, dim);
		cudaCheckErrors();
		
		//gauss
		gauss<<<gauss_grid, gauss_block>>>(d_A, d_I, iter, dim);
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_A, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		print_matrix(h_A, h_I, dim);
		cudaCheckErrors();
	}
	cudaCheckErrors();
}*/


void cuda_offload(float *matrix, float *iden, int dim) {
	float *d_A, *d_I;
	
	// setup and copy matrices to gpu
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_I, dim * dim * sizeof(float));
	cudaMemcpy(d_A, matrix, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, iden, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors();
	
	// setup kernelsizes
	
	int rowp = 3;
	
	int threadsperblock = 128;
	int blocks = std::ceil((dim + 1.) / threadsperblock);
	
	dim3 norm_block(threadsperblock);
	dim3 norm_grid(blocks);
	
	
	dim3 gauss_block(threadsperblock);
	dim3 gauss_grid(1, dim - 1);
	
	blocks = std::ceil(static_cast<float>(dim) / threadsperblock);
	dim3 gauss_fix_block(1, threadsperblock);
	dim3 gauss_fix_grid(1, blocks);
	
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
		//cudaDeviceSynchronize();
		cudaMemcpy(&matrix[iter * dim + iter], &d_A[iter * dim + iter], sizeof(float), cudaMemcpyDeviceToHost);
		if (matrix[iter * dim + iter] == 0.0f) {
			printf("error at iter %d\n", iter);
			break;
		}
		//normalize
		//cudaMemcpy(d_I, h_I, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_A, h_A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
		normalize<<<norm_grid, norm_block>>>(d_A, d_I, iter, dim, matrix[iter * dim + iter]);
		cudaDeviceSynchronize();
		/*cudaMemcpy(iden, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrix, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		print_matrix(matrix, iden, dim);*/
		cudaCheckErrors();
		
		//gauss
		//cudaMemcpy(d_I, h_I, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_A, h_A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
		gauss<<<gauss_grid, gauss_block>>>(d_A, d_I, iter, dim);
		cudaCheckErrors();
		gauss_fix<<<gauss_fix_grid, gauss_fix_block>>>(d_A, iter, dim);
		cudaDeviceSynchronize();
		/*cudaMemcpy(iden, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrix, d_A, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
		print_matrix(matrix, iden, dim);*/
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
	//print_matrix(matrix, iden, dim);
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
	printf("\n");*/
}
