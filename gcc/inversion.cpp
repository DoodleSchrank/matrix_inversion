#include <omp.h>

#define minvalue 0.0000

template<int dim>
void single_cpu(float matrix[dim][dim], float iden[dim][dim]) {
	float factor;
	for (int i = 0; i < dim; i++) {
		if (matrix[i][i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j][i] != 0) {
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i][x] += matrix[j][x];
						iden[i][x] = iden[j][x];
					}
					break;
				}
			}
		}
		
		//normalize
		factor = matrix[i][i];
		for (int x = 0; x < 2 * dim; x++) {
			if (x < dim)
				matrix[i][x] /= factor;
			else
				iden[i][x - dim] /= factor;
		}
		
		//gauss
		for (int y = 0; y < dim; y++) {
			factor = matrix[y][i];
			if (y != i && factor != 0.0f) {
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y][x] -= matrix[i][x] * factor;
					else
						iden[y][x - dim] -= iden[i][x - dim] * factor;
				}
			}
		}
	}
}


template<int dim>
void openmp_offload(float matrix[dim][dim], float iden[dim][dim]) {
	float factor;
#pragma omp target enter data map(to: matrix[0:dim][0:dim], iden[0:dim][0:dim]) map(alloc: factor)
	for (int i = 0; i < dim; i++) {
		if (matrix[i][i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j][i] != 0) {
#pragma omp target teams distribute parallel for simd
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i][x] += matrix[j][x];
						iden[i][x] = iden[j][x];
					}
					break;
				}
			}
		}
		
		
		//normalize
#pragma omp target update from(matrix[i][i])
		factor = matrix[i][i];
#pragma omp target teams distribute shared(factor)
		for (int x = 0; x < 2 * dim; x++) {
			if (x < dim)
				matrix[i][x] /= factor;
			else
				iden[i][x - dim] /= factor;
		}
		
		//gauss
#pragma omp target teams distribute parallel for private(factor)
		for (int y = 0; y < dim; y++) {
			factor = matrix[y][i];
			if (y != i && factor != 0.0f) {
#pragma omp simd
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y][x] -= matrix[i][x] * factor;
					else
						iden[y][x - dim] -= iden[i][x - dim] * factor;
				}
			}
		}
	}
	/*/normalize
#pragma omp target teams distribute
	for (int i = 0; i < dim; i++) {
		factor = matrix[i][i];
		matrix[i][i] /= factor;
		for(int x = 0; x < dim; x++) {
			iden[i][x] /= factor;
		}
	}*/
#pragma omp target exit data map(from: iden[0:dim][0:dim])
}


template<int dim>
void openacc_offload(float matrix[dim][dim], float iden[dim][dim]) {
	float factor;
#pragma acc enter data copyin(matrix[0:dim][0:dim], iden[0:dim][0:dim]), create(factor)
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
#pragma acc serial
		{
			factor = matrix[i][i];
		};

//#pragma acc host_data use_device(factor)
#pragma acc parallel loop
		for (int x = i; x < dim + i + 1; x++) {
			if (x < dim)
				matrix[i][x] /= factor;
			else
				iden[i][x - dim] /= factor;
		}
		
		//gauss
#pragma acc parallel loop gang worker //vector_length(32)
		for (int y = 0; y < dim; y++) {
			factor = matrix[y][i];
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

/*
template<int dim>
void cuda_offload(float matrix[dim][dim], float iden[dim][dim]) {
	auto h_A = reinterpret_cast<float[dim * dim]>(matrix);
	auto h_I = reinterpret_cast<float[dim * dim]>(iden);
	float *d_A, *d_I;
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_I, dim * dim * sizeof(float));
	cudaCheckErrors("cudaMalloc failure");
	cudaMemcpy(d_A, h_A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy H2D failure");
	
	// Launch kernel
	int block_size = (int) min(2048, dim);
	dim3 inv_block(block_size);
	dim3 inv_grid((int) ceil(dim / block_size));
	dim3 gauss_block(sqr(block_size), sqr(block_size));
	dim3 gauss_grid((int) ceil(dim / gauss_block.x), (int) ceil(dim / gauss_block.y));
	
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
		normalize<<<inv_grid, inv_block>>>(d_A, d_I, iter, dim);
		
		//gauss
		gauss<<<gauss_grid, gauss_block>>>(d_A, d_I, iter, dim);
	}
	cudaCheckErrors("kernel launch failure");
	
	// CUDA processing sequence step 2 is complete
	
	// Copy results back to host
	cudaMemcpy(h_I, d_I, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void normalize(float *matrix, float *iden, int iter, int dim) {
	int x = iter + blockDim.x * blockIdx.x + threadIdx.x;
	float factor = matrix[iter * dim + iter];
	if (x < dim)
		matrix[iter * dim + x] /= factor;
	else if(x < dim + iter + 1)
		iden[iter * dim + x - dim] /= factor;
}

__global void gauss(float *matrix, float *iden, int iter, int dim) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	if (x > dim + iter || y >= dim)
		return;
	
	float factor = matrix[y * dim + iter];
	if (y != iter && factor != 0.0f) {
		if (x < dim)
			matrix[y * dim + x] -= matrix[iter * dim + x] * factor;
		else
			iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;
	}
}
*/