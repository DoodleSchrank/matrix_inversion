#include "cublas_v2.h"
#include "cuda_runtime.h"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif


#define cudacall(call)                                                                                                        \
	do {                                                                                                                      \
		cudaError_t err = (call);                                                                                             \
		if (cudaSuccess != err) {                                                                                             \
			fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			cudaDeviceReset();                                                                                                \
			exit(EXIT_FAILURE);                                                                                               \
		}                                                                                                                     \
	} while (0)

#define cublascall(call)                                                                                     \
	do {                                                                                                     \
		cublasStatus_t status = (call);                                                                      \
		if (CUBLAS_STATUS_SUCCESS != status) {                                                               \
			fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status); \
			cudaDeviceReset();                                                                               \
			exit(EXIT_FAILURE);                                                                              \
		}                                                                                                    \
                                                                                                             \
	} while (0)

void cublas_offload(scalar *matrix, scalar *result, int dim) {
	scalar **inputs = (scalar **) new scalar *;
	auto **results = (scalar **) new scalar *;
	scalar **d_results;
	scalar *d_result;
	
	cudacall(cudaMalloc(&d_results, sizeof(float *)));
	cudacall(cudaMalloc(&d_result, dim * dim * sizeof(float)));
	results[0] = d_result;
	cudacall(cudaMemcpy(d_results, results, sizeof(float *), cudaMemcpyHostToDevice));
	
	cublasHandle_t cu_handle;
	cublascall(cublasCreate_v2(&cu_handle));

	int *pivot_element;
	int h_info[1];
	int *d_info;

	cudacall(cudaMalloc(&pivot_element, sizeof(int)));
	cudacall(cudaMalloc(&d_info, sizeof(int)));

	auto **matrices = (scalar **) new scalar *;
	scalar **d_matrices;
	scalar *d_matrix;

	cudacall(cudaMalloc(&d_matrices, sizeof(scalar *)));
	cudacall(cudaMalloc(&d_matrix, dim * dim * sizeof(scalar)));

	matrices[0] = d_matrix;

	cudacall(cudaMemcpy(d_matrices, matrices, sizeof(scalar *), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_matrix, matrix, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice));

	cublascall(cublasSgetrfBatched(cu_handle, dim, d_matrices, dim, pivot_element, d_info, 1));
	cudacall(cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
	if (h_info[0] != 0) {
		fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", 0);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	float **C = (float **) new float *;
	float **C_d, *C_dflat;

	cudacall(cudaMalloc(&C_d, sizeof(float *)));
	cudacall(cudaMalloc(&C_dflat, dim * dim * sizeof(float)));
	C[0] = C_dflat;
	cudacall(cudaMemcpy(C_d, C, sizeof(float *), cudaMemcpyHostToDevice));


	cublascall(cublasSgetriBatched(cu_handle, dim, (const scalar **) d_matrices, dim, pivot_element, d_results, dim, d_info, 1));
	cudacall(cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
	if (h_info[0] != 0) {
		fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", 0);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
	cudacall(cudaMemcpy(result, d_result, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost));

	cudaFree(d_matrices);
	cudaFree(d_matrix);
	cudaFree(d_result);
	cudaFree(d_results);
	free(matrices);
	free(C);
	cudaFree(pivot_element);
	cudaFree(d_info);
	cublasDestroy_v2(cu_handle);
}