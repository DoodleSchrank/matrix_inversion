#include <omp.h>
#include <stdio.h>
#include <array>
//#include <CL/opencl.h>
#include <stdlib.h>
#include <err.h>

void print_matrix(float *matrix, float *iden, int dim) {
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
/*#define CaseReturnString(x) case x: return #x;

char *opencl_errstr(cl_int err)
{
	switch (err)
	{
		CaseReturnString(CL_SUCCESS                        )
		CaseReturnString(CL_DEVICE_NOT_FOUND               )
		CaseReturnString(CL_DEVICE_NOT_AVAILABLE           )
		CaseReturnString(CL_COMPILER_NOT_AVAILABLE         )
		CaseReturnString(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
		CaseReturnString(CL_OUT_OF_RESOURCES               )
		CaseReturnString(CL_OUT_OF_HOST_MEMORY             )
		CaseReturnString(CL_PROFILING_INFO_NOT_AVAILABLE   )
		CaseReturnString(CL_MEM_COPY_OVERLAP               )
		CaseReturnString(CL_IMAGE_FORMAT_MISMATCH          )
		CaseReturnString(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
		CaseReturnString(CL_BUILD_PROGRAM_FAILURE          )
		CaseReturnString(CL_MAP_FAILURE                    )
		CaseReturnString(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
		CaseReturnString(CL_COMPILE_PROGRAM_FAILURE        )
		CaseReturnString(CL_LINKER_NOT_AVAILABLE           )
		CaseReturnString(CL_LINK_PROGRAM_FAILURE           )
		CaseReturnString(CL_DEVICE_PARTITION_FAILED        )
		CaseReturnString(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
		CaseReturnString(CL_INVALID_VALUE                  )
		CaseReturnString(CL_INVALID_DEVICE_TYPE            )
		CaseReturnString(CL_INVALID_PLATFORM               )
		CaseReturnString(CL_INVALID_DEVICE                 )
		CaseReturnString(CL_INVALID_CONTEXT                )
		CaseReturnString(CL_INVALID_QUEUE_PROPERTIES       )
		CaseReturnString(CL_INVALID_COMMAND_QUEUE          )
		CaseReturnString(CL_INVALID_HOST_PTR               )
		CaseReturnString(CL_INVALID_MEM_OBJECT             )
		CaseReturnString(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
		CaseReturnString(CL_INVALID_IMAGE_SIZE             )
		CaseReturnString(CL_INVALID_SAMPLER                )
		CaseReturnString(CL_INVALID_BINARY                 )
		CaseReturnString(CL_INVALID_BUILD_OPTIONS          )
		CaseReturnString(CL_INVALID_PROGRAM                )
		CaseReturnString(CL_INVALID_PROGRAM_EXECUTABLE     )
		CaseReturnString(CL_INVALID_KERNEL_NAME            )
		CaseReturnString(CL_INVALID_KERNEL_DEFINITION      )
		CaseReturnString(CL_INVALID_KERNEL                 )
		CaseReturnString(CL_INVALID_ARG_INDEX              )
		CaseReturnString(CL_INVALID_ARG_VALUE              )
		CaseReturnString(CL_INVALID_ARG_SIZE               )
		CaseReturnString(CL_INVALID_KERNEL_ARGS            )
		CaseReturnString(CL_INVALID_WORK_DIMENSION         )
		CaseReturnString(CL_INVALID_WORK_GROUP_SIZE        )
		CaseReturnString(CL_INVALID_WORK_ITEM_SIZE         )
		CaseReturnString(CL_INVALID_GLOBAL_OFFSET          )
		CaseReturnString(CL_INVALID_EVENT_WAIT_LIST        )
		CaseReturnString(CL_INVALID_EVENT                  )
		CaseReturnString(CL_INVALID_OPERATION              )
		CaseReturnString(CL_INVALID_GL_OBJECT              )
		CaseReturnString(CL_INVALID_BUFFER_SIZE            )
		CaseReturnString(CL_INVALID_MIP_LEVEL              )
		CaseReturnString(CL_INVALID_GLOBAL_WORK_SIZE       )
		CaseReturnString(CL_INVALID_PROPERTY               )
		CaseReturnString(CL_INVALID_IMAGE_DESCRIPTOR       )
		CaseReturnString(CL_INVALID_COMPILER_OPTIONS       )
		CaseReturnString(CL_INVALID_LINKER_OPTIONS         )
		CaseReturnString(CL_INVALID_DEVICE_PARTITION_COUNT )
		default: return "Unknown OpenCL error code";
	}
}*/


void single_cpu(float *matrix, float *iden, int dim) {
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
				for (int x = i; x < dim; x++) { // swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
				
			}
		}
		
		//normalize
		float factor = matrix[i * dim + i];
		for (int x = 0; x < 2 * dim; x++) {
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else
				iden[i * dim + x - dim] /= factor;
		}
		
		//gauss
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
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


void openmp_offload(float *matrix, float *iden, int dim) {
	int i;
#pragma omp target data map(tofrom: matrix[0:dim*dim], iden[0:dim*dim]) map(alloc: i)
	for (i = 0; i < dim; i++) {
//#pragma omp target update to(i)
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
#pragma omp target teams distribute parallel for simd
				for (int x = i; x < dim; x++) { // swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
			}
		}
		
		//normalize
#pragma omp target teams distribute parallel for
		for (int x = i + 1; x < dim + i + 1; x++) {
			float factor = matrix[i * dim + i];
			if (x < dim) {
				matrix[i * dim + x] /= factor;
			} else {
				iden[i * dim + x - dim] /= factor;
			}
		}
		matrix[i * dim + i] = 1;
#pragma omp target update to(matrix[i * dim + i])
		
		//gauss
#pragma omp target teams distribute parallel for
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
#pragma omp simd
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y * dim + x] -= matrix[i * dim + x] * factor;
					else
						iden[y * dim + x - dim] -= iden[i * dim + x - dim] * factor;
				}
			}
		}
	}
//#pragma omp target exit data map(from: matrix[0:dim*dim], iden[0:dim*dim])
}


/*#define CL_TARGET_OPENCL_VERSION 300

const char *normalize_str =
		"__kernel void normalize_matrix(\n"
		"__global float *matrix,\n"
		"__global float *iden,\n"
		"int dim,\n"
		"int iter) {\n"
		"int x = get_global_id(0);\n"
		"if (x >= 2 * dim){\n"
		"   return;}\n"
		"if (x < dim){\n"
		"   matrix[iter * dim + x] /= matrix[iter * dim + iter];\n"
		"} else {\n"
		"   iden[iter * dim + x - dim] /= matrix[iter * dim + iter];}\n"
		"}";

const char *gauss_str =
		"__kernel void gauss(\n"
		"		__global float *matrix,\n"
		"		__global float *iden,\n"
		"		int dim,\n"
		"		int iter) {\n"
		"int x = get_global_id(0);\n"
		
		"int y = get_global_id(1);\n"
		"if (x >= 2 * dim || y == iter)\n"
		"return;\n"
		
		"float factor = matrix[y * dim + iter];\n"
		
		"if (x < dim)\n"
		"matrix[y * dim + x] -= matrix[iter * dim + x] * factor;\n"
		"else\n"
		"iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;\n"
		"}";

const char *gaussfix_str =
		"__kernel void gaussfix(\n"
		"		__global float *matrix,\n"
		"		__global float *iden,\n"
		"		int dim,\n"
		"		int iter) {\n"
		"int x = get_global_id(0);\n"
		
		"if (x >= dim || x == iter)\n"
		"return;\n"
		
		"matrix[x * dim + iter] = 0;\n"
		"}";


void opencl_offload(float *matrix, float *iden, int dim) {
	cl_int errCode;
	
	// Obtain the first available platform.
	cl_platform_id platformID = NULL;
	cl_uint numPlatforms;
	errCode = clGetPlatformIDs(1, &platformID, &numPlatforms);
	if (errCode != CL_SUCCESS) {
		errx(1, "clGetPlatformIDs() failed");
	}
	
	// Obtain the first available device on the platform
	cl_device_id deviceID = NULL;
	cl_uint numDevices;
	errCode = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1,
	                         &deviceID, &numDevices);
	if (errCode != CL_SUCCESS) {
		errx(1, "clGetDeviceIDs() failed");
	}
	
	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateContext() failed");
	}
	
	// Create a command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateCommandQueue() failed");
	}
	size_t iter = 0;
	// Allocate memory on the device
	cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(float), NULL, &errCode);
	cl_mem d_I = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(float), NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateBuffer() failed");
	}
	
	// Copy data from the host to the device
	errCode = clEnqueueWriteBuffer(commandQueue, d_A, CL_FALSE, 0, dim * dim * sizeof(float), matrix, 0, NULL, NULL);
	errCode |= clEnqueueWriteBuffer(commandQueue, d_I, CL_FALSE, 0, dim * dim * sizeof(float), iden, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clEnqueueWriteBuffer() failed");
	}
	
	//
	// Compute on the device
	//
	
	// Creates a program object for a context, and loads source code specified by text strings into the program object
	cl_program normalize_program = clCreateProgramWithSource(context, 1, &normalize_str, NULL, &errCode);
	cl_program gauss_program = clCreateProgramWithSource(context, 1, &gauss_str, NULL, &errCode);
	cl_program gaussfix_program = clCreateProgramWithSource(context, 1, &gaussfix_str, NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateProgramWithSource() failed");
	}
	
	// Builds (compiles and links) a program executable from the program source
	errCode = clBuildProgram(normalize_program, 1, &deviceID, NULL, NULL, NULL);
	errCode |= clBuildProgram(gauss_program, 1, &deviceID, NULL, NULL, NULL);
	errCode |= clBuildProgram(gaussfix_program, 1, &deviceID, NULL, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(normalize_program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		errx(1, "clBuildProgram() gaussfix failed:\n%s", buffer);
	}
	
	// Creates a kernel object
	cl_kernel normalize_kernel = clCreateKernel(normalize_program, "normalize_matrix", &errCode);
	cl_kernel gauss_kernel = clCreateKernel(gauss_program, "gauss", &errCode);
	cl_kernel gaussfix_kernel = clCreateKernel(gaussfix_program, "gaussfix", &errCode);
	if (errCode != CL_SUCCESS) {
		printf("error: %d,%d\n", errCode, CL_INVALID_KERNEL_NAME);
		errx(1, "clCreateKernel() failed");
	}
	
	// Set the argument value for a specific argument of a kernel
	errCode = clSetKernelArg(normalize_kernel, 0, sizeof(cl_mem), &d_A);
	errCode |= clSetKernelArg(normalize_kernel, 1, sizeof(cl_mem), &d_I);
	errCode |= clSetKernelArg(normalize_kernel, 2, sizeof(int), &dim);
	
	errCode |= clSetKernelArg(gauss_kernel, 0, sizeof(cl_mem), &d_A);
	errCode |= clSetKernelArg(gauss_kernel, 1, sizeof(cl_mem), &d_I);
	errCode |= clSetKernelArg(gauss_kernel, 2, sizeof(int), &dim);
	
	errCode |= clSetKernelArg(gaussfix_kernel, 0, sizeof(cl_mem), &d_A);
	errCode |= clSetKernelArg(gaussfix_kernel, 1, sizeof(cl_mem), &d_I);
	errCode |= clSetKernelArg(gaussfix_kernel, 2, sizeof(int), &dim);
	if (errCode != CL_SUCCESS) {
		errx(1, "clSetKernelArg() failed");
	}
	
	
	// Enqueues a command to execute a kernel on a device
	size_t normalize_threads = ceil(static_cast<size_t>(dim) + 1);
	size_t gauss_threads[2] = {2 * static_cast<size_t>(dim), static_cast<size_t>(dim) - 1};
	
	float one[1] = {1.};
	
	for (iter = 0; iter < dim; iter++) {
		const size_t *itr = &iter;
		errCode |= clSetKernelArg(normalize_kernel, 3, sizeof(int), &iter);
		errCode |= clSetKernelArg(gauss_kernel, 3, sizeof(int), &iter);
		errCode |= clSetKernelArg(gaussfix_kernel, 3, sizeof(int), &iter);
		
		errCode = clEnqueueNDRangeKernel(commandQueue, normalize_kernel, 1, itr, &normalize_threads, NULL, 0, NULL,
		                                 NULL);
		errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_FALSE, 0, dim * dim * sizeof(float), matrix, 0, NULL, NULL);
		errCode |= clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(float), iden, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("read after normalization failed %s\n", opencl_errstr(errCode));
		}
		
		
		//errCode |= clEnqueueWriteBuffer(commandQueue, d_A, CL_TRUE, (iter * dim + iter) * sizeof(float), sizeof(float), one, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after normalize read%s\n", opencl_errstr(errCode));
		}
		print_matrix(matrix, iden, dim);
		
		
		errCode = clEnqueueNDRangeKernel(commandQueue, gauss_kernel, 2, itr, gauss_threads, NULL, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gauss %s\n", opencl_errstr(errCode));
		}
		errCode |= clEnqueueNDRangeKernel(commandQueue, gaussfix_kernel, 1, NULL, &normalize_threads, NULL, 0, NULL,NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gaussfix %s\n", opencl_errstr(errCode));
		}
		errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_FALSE, 0, dim * dim * sizeof(float), matrix, 0, NULL, NULL);
		errCode |= clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(float), iden, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gauss read %s\n", opencl_errstr(errCode));
		}
		print_matrix(matrix, iden, dim);
	}
	// Wait for command completion
	errCode = clFinish(commandQueue);
	if (errCode != CL_SUCCESS) {
		errx(1, "clFinish() failed");
	}
	
	// Release the kernel object
	errCode = clReleaseKernel(normalize_kernel);
	errCode = clReleaseKernel(gauss_kernel);
	errCode = clReleaseKernel(gaussfix_kernel);
	
	// Release the program object
	errCode = clReleaseProgram(normalize_program);
	errCode = clReleaseProgram(gauss_program);
	errCode = clReleaseProgram(gaussfix_program);
	
	// Release the device
	errCode = clReleaseDevice(deviceID);
	
	// Transfer data back from the device to the host
	errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_TRUE, 0, dim * dim * sizeof(float), matrix, 0, NULL, NULL);
	errCode |= clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(float), iden, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clEnqueueReadBuffer() failed");
	}
	
	// Delete data on the device
	errCode = clReleaseMemObject(d_A);
	errCode |= clReleaseMemObject(d_I);
	if (errCode != CL_SUCCESS) {
		errx(1, "clReleaseMemObject() failed");
	}
	
	// Release a command queue
	errCode = clReleaseCommandQueue(commandQueue);
	
	// release the context
	errCode = clReleaseContext(context);
	
	
	
	
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
}*/
