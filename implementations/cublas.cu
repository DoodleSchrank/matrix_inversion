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

void cublas_offload(scalar *A, scalar *I, int dim) {
	auto **As = (scalar **) new scalar *;
	auto **Is = (scalar **) new scalar *;
	scalar **d_As;
	scalar **d_Is;
	scalar *d_A;
	scalar *d_I;

	cudacall(cudaMalloc(&d_As, sizeof(scalar *)));
	cudacall(cudaMalloc(&d_Is, sizeof(scalar *)));
	cudacall(cudaMalloc(&d_A, dim * dim * sizeof(scalar)));
	cudacall(cudaMalloc(&d_I, dim * dim * sizeof(scalar)));
	As[0] = d_A;
	Is[0] = d_I;
	cudacall(cudaMemcpy(d_As, As, sizeof(scalar *), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_Is, Is, sizeof(scalar *), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_A, A, dim * dim * sizeof(scalar), cudaMemcpyHostToDevice));


	cublasHandle_t cu_handle;
	cublascall(cublasCreate_v2(&cu_handle));
	int *pivot_element;
	int *d_info;
	cudacall(cudaMalloc(&pivot_element, sizeof(int)));
	cudacall(cudaMalloc(&d_info, sizeof(int)));

#ifdef dbl
	cublascall(cublasDgetrfBatched(cu_handle, dim, d_As, dim, pivot_element, d_info, 1));
#else
	cublascall(cublasSgetrfBatched(cu_handle, dim, d_As, dim, pivot_element, d_info, 1));
#endif

#ifdef dbl
	cublascall(cublasDgetriBatched(cu_handle, dim, (const scalar **) d_As, dim, pivot_element, d_Is, dim, d_info, 1));
#else
	cublascall(cublasSgetriBatched(cu_handle, dim, (const scalar **) d_As, dim, pivot_element, d_Is, dim, d_info, 1));
#endif
	cudacall(cudaMemcpy(I, d_I, dim * dim * sizeof(scalar), cudaMemcpyDeviceToHost));

	cudaFree(d_As);
	cudaFree(d_A);
	cudaFree(d_I);
	cudaFree(d_Is);
	free(As);
	cudaFree(pivot_element);
	cudaFree(d_info);
	cublasDestroy_v2(cu_handle);
}
