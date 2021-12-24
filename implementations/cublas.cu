#include "cublas_v2.h"
#include "cuda_runtime.h"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

#define cudaCheckErrors()                                                                    \
	{                                                                                        \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e != cudaSuccess) {                                                              \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0);                                                                         \
		}                                                                                    \
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
	scalar *const *matrices = &matrix;
	scalar *const *identities = &iden;

	cublasSgetrfBatched(cu_handle, dim, matrices, dim, NULL, info, 1);
	cublasSgetriBatched(cu_handle, dim, matrices, dim, NULL, identities, dim, info, 1);
	cudaCheckErrors();
	cudaMemcpy(iden, d_I, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrix, d_A, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_I);
	cudaCheckErrors();
}
