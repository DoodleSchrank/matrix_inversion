#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "inversion.cu"
#include <cuda_runtime.h>
#include <chrono>
#include <string>

void matrix_read(int dim, float* matrix) {
	int row = 0;
	
	std::ifstream infile(("../randomMatrix_" + std::to_string(dim) + ".txt").c_str());
	for (std::string line; std::getline(infile, line);) {
		std::istringstream inputline(line);
		for (int i = 0; i < dim; i++) {
			inputline >> matrix[row * dim + i];
		}
		row++;
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2 || argc > 3)
		return 0;
	int dimension = std::stoi(argv[1]);
	auto matrix = new float[dimension * dimension];
	auto identity_matrix = new float[dimension * dimension];
	matrix_read(dimension, static_cast<float*>(matrix));
	
	
	//fill identity matrix
#pragma omp parallel for collapse(2)
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			if (i == j) {
				identity_matrix[i * dimension + i] = 1;
			} else {
				identity_matrix[i * dimension + j] = 0;
			}
		}
	}
	
	
	
	/*auto openacc = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &openacc[0 * dimension + 0]);
	auto openaccres = new float[dimension * dimension + dimension];
	std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &openaccres[0 * dimension + 0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	openacc_offload(openacc, openaccres, dimension);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openacc_offload_time = end - start;
	printf("%04f\n", openacc_offload_time.count());*/
	//printf("OpenACC offload Time: %04f\n", openacc_offload_time.count());
	
	
	
	auto cuda = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &cuda[0 * dimension + 0]);
	auto cudares = new float[dimension * dimension + dimension];
	std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &cudares[0 * dimension + 0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	cuda_offload(cuda, cudares, dimension);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> cuda_time = end - start;
	printf("%04f\n", cuda_time.count());
	//printf("CUDA Time: %04f\n", cuda_time.count());
	
	
	/*float threshold = 0.00001;
	int errc = 0;
	float minerr = 1000000.;
	float maxerr = threshold;
	float maxerrval;
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			float error = fabs(openmpres[i][j] - cudares[i][j]);
			if (error > threshold || std::isnan(error)) {
				errc++;
			}
			if (maxerr < error) {
				maxerr = error;
				maxerrval = openmpres[i][j];
			}
			if (minerr > error) minerr = error;
		}
	}
	std::cout << "errc: " << errc << std::endl;
	std::cout << "maxerrval: " << maxerrval << " maxerr: " << maxerr << "  minerr: " << minerr << std::endl;*/
	
	
	/*for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			printf("%2f ", openmpres[i][j]);
		}
		printf("      ");
		for (int j = 0; j < dimension; j++) {
			printf("%2f ", cudares[i][j]);
		}
		printf("\n");
	}*/
	
	return 0;
}
