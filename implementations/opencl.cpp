#include <CL/opencl.h>
#include <err.h>
#include <string>

#ifdef dbl
	using scalar = double;
#else
using scalar = float;
#endif




#define CaseReturnString(x) case x: return #x;

const char* opencl_errstr(cl_int err)
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
}

#define CL_TARGET_OPENCL_VERSION 300

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


void opencl_offload(scalar *matrix, scalar *iden, int dim) {
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
	cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(scalar), NULL, &errCode);
	cl_mem d_I = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(scalar), NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateBuffer() failed");
	}
	
	// Copy data from the host to the device
	errCode = clEnqueueWriteBuffer(commandQueue, d_A, CL_FALSE, 0, dim * dim * sizeof(scalar), matrix, 0, NULL, NULL);
	errCode |= clEnqueueWriteBuffer(commandQueue, d_I, CL_FALSE, 0, dim * dim * sizeof(scalar), iden, 0, NULL, NULL);
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
	
	scalar one[1] = {1.};
	
	for (iter = 0; iter < dim; iter++) {
		const size_t *itr = &iter;
		errCode |= clSetKernelArg(normalize_kernel, 3, sizeof(int), &iter);
		errCode |= clSetKernelArg(gauss_kernel, 3, sizeof(int), &iter);
		errCode |= clSetKernelArg(gaussfix_kernel, 3, sizeof(int), &iter);
		
		errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_FALSE, 0, dim * dim * sizeof(scalar), matrix, 0, NULL, NULL);
		errCode |= clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(scalar), iden, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("read before normalization failed %s\n", opencl_errstr(errCode));
		}
		
		
		errCode = clEnqueueNDRangeKernel(commandQueue, normalize_kernel, 1, itr, &normalize_threads, NULL, 0, NULL,
										 NULL);
		errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_TRUE, 0, dim * dim * sizeof(scalar), matrix, 0, NULL, NULL);
		errCode |= clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(scalar), iden, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("read after normalization failed %s\n", opencl_errstr(errCode));
		}
		
		errCode |= clEnqueueWriteBuffer(commandQueue, d_A, CL_TRUE, (iter * dim + iter) * sizeof(scalar), sizeof(scalar), one, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after normalize read%s\n", opencl_errstr(errCode));
		}
		
		errCode = clEnqueueNDRangeKernel(commandQueue, gauss_kernel, 2, itr, gauss_threads, NULL, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gauss %s\n", opencl_errstr(errCode));
		}
		errCode |= clEnqueueNDRangeKernel(commandQueue, gaussfix_kernel, 1, NULL, &normalize_threads, NULL, 0, NULL,NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gaussfix %s\n", opencl_errstr(errCode));
		}
		errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_TRUE, 0, dim * dim * sizeof(scalar), matrix, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gauss read 1 %s\n", opencl_errstr(errCode));
		}
		errCode = clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(scalar), iden, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			printf("after gauss read 2 %s\n", opencl_errstr(errCode));
		}
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
	errCode = clEnqueueReadBuffer(commandQueue, d_A, CL_TRUE, 0, dim * dim * sizeof(scalar), matrix, 0, NULL, NULL);
	errCode |= clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(scalar), iden, 0, NULL, NULL);
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
}
