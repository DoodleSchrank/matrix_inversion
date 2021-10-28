#include <omp.h>
#include <stdio.h>
#include <array>

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

int bound(int coord) {
	if (coord > 2) return 0;
	
	if (coord < 0) return 2;
	
	return coord;
}

float det3x3(float **matrix) {
	float det = 0;
	float diagonal = 0;
	int x = 0, y = 0;
	
	for (int i = 0; i < 3; i++) {
		diagonal = matrix[x][y];
		for (int j = 1; j < 3; j++) {
			x = bound(x++);
			y = bound(y++);
			diagonal *= matrix[x][y];
		}
		x = bound(x++);
		det += diagonal;
	}
	
	for (int i = 0; i < 3; i++) {
		diagonal = matrix[x][y];
		
		for (int j = 1; j < 3; j++) {
			x = bound(x--);
			y = bound(y++);
			diagonal *= matrix[x][y];
		}
		x = bound(x--);
		det -= diagonal;
	}
	return det;
}


void adjugate(float **matrix) {
	std::array<float[3][3], 16> subarrays;
	
	int overx = 0;
	int overy = 0;
	
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			auto arr = subarrays.at(y * dim + x);
			for (int yi = 0; yi < 4; yi++) {
				if (yi == y) {
					overy--;
					continue;
				}
				for (int xi = 0; xi < 4; xi++) {
					if (xi == x) {
						overx--;
						continue;
					}
					arr[xi + overx][yi + overy] = matrix[xi][yi];
				}
				overx++;
			}
			overy++;
		}
	}
	
	
	float det = matrix[0][0] *
	            det3x3(subarrays.at(0)) - matrix[1][0] *
	            det3x3(subarrays.at(1)) + matrix[2][0] *
	            det3x3(subarrays.at(2)) - matrix[3][0] *
	            det3x3(subarrays.at(3));
	float subdet[16];
	int i = 0;
	for (float **matrix : subarrays) {
		i++;
		matrix[i / 4][i % 4] = (i % 2) ? det / det3x3(matrix) : -det / det3x3(matrix);
	}
}

//const char *normalize =
__kernel void inc(
		__global float *d_A,
		__global float *d_I,
		int dim,
		int iter) {
	int i = get_global_id(0);
	int row = i / dim;
	int x = i - row * dim;
	if (row == iter && x >= iter && x < dim + iter + 1) {
		if (x < dim) {
			d_A[i] = d_A[i] / d_A[row * dim + iter];
		} else {
			d_I[i] = d_I[i] / d_A[row * dim + iter];
		}
	}
}


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
	errCode = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1,
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
	
	// Allocate memory on the device
	cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(float), NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateBuffer() failed");
	}
	cl_mem d_I = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(float), NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateBuffer() failed");
	}
	
	// Copy data from the host to the device
	errCode = clEnqueueWriteBuffer(commandQueue, d_A, CL_TRUE, 0, dim * dim * sizeof(float), matrix, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clEnqueueWriteBuffer() failed");
	}
	// Copy data from the host to the device
	errCode = clEnqueueWriteBuffer(commandQueue, d_I, CL_TRUE, 0, dim * dim * sizeof(float), iden, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clEnqueueWriteBuffer() failed");
	}
	
	//
	// Compute on the device
	//
	
	// Creates a program object for a context, and loads source code specified by text strings into the program object
	cl_program program = clCreateProgramWithSource(context, 1, &incSource, NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateProgramWithSource() failed");
	}
	
	// Builds (compiles and links) a program executable from the program source
	errCode = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		errx(1, "clBuildProgram() failed:\n%s", buffer);
	}
	
	// Creates a kernel object
	cl_kernel kernel = clCreateKernel(program, "inc", &errCode);
	if (errCode != CL_SUCCESS) {
		errx(1, "clCreateKernel() failed");
	}
	
	// Set the argument value for a specific argument of a kernel
	errCode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
	if (errCode != CL_SUCCESS) {
		errx(1, "clSetKernelArg() failed");
	}
	errCode = clSetKernelArg(kernel, 1, sizeof(unsigned int), &size);
	if (errCode != CL_SUCCESS) {
		errx(1, "clSetKernelArg() failed");
	}
	
	// Query the maximum workgroup size
	size_t local;
	errCode = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clGetKernelWorkGroupInfo() failed");
	}
	
	// Enqueues a command to execute a kernel on a device
	size_t global = size;
	errCode = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clEnqueueNDRangeKernel() failed");
	}
	
	// Wait for command completion
	errCode = clFinish(commandQueue);
	if (errCode != CL_SUCCESS) {
		errx(1, "clFinish() failed");
	}
	
	// Release the kernel object
	errCode = clReleaseKernel(kernel);
	
	// Release the program object
	errCode = clReleaseProgram(program);
	
	// Release the device
	errCode = clReleaseDevice(deviceID);
	
	// Transfer data back from the device to the host
	errCode = clEnqueueReadBuffer(commandQueue, a_d, CL_TRUE, 0, size * sizeof(double), a, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		errx(1, "clEnqueueReadBuffer() failed");
	}
	
	// Delete data on the device
	errCode = clReleaseMemObject(a_d);
	if (errCode != CL_SUCCESS) {
		errx(1, "clReleaseMemObject() failed");
	}
	
	// Release a command queue
	errCode = clReleaseCommandQueue(commandQueue);
	
	// release the context
	errCode = clReleaseContext(context);
	
	// Postprocess data on the host
	// e.g. write data to storage
	for (int i = 0; i < size; i++) {
		if (a[i] != 1.) {
			errx(2, "Computation on GPU failed");
		}
	}
	
	// Free memory on the host
	free(a);
	
	return 0;
}

