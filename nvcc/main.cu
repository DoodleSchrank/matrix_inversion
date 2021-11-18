#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "inversion.cu"
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <algorithm>

void matrix_read(int dim, float *matrix) {
	int row = 0;
	
	std::ifstream infile(("randomMatrix_" + std::to_string(dim) + ".txt").c_str());
	for (std::string line; std::getline(infile, line);) {
		std::istringstream inputline(line);
		for (int i = 0; i < dim; i++) {
			inputline >> matrix[row * dim + i];
		}
		row++;
	}
}

int main(int argc, char *argv[]) {
	int dimension = std::stoi(argv[1]);
	int algorithms = std::stoi(argv[2]);
	int run = std::stoi(argv[3]);
	auto matrix = new float[dimension * dimension];
	auto identity_matrix = new float[dimension * dimension];
	matrix_read(dimension, static_cast<float *>(matrix));
	
	
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
	
	std::chrono::time_point <std::chrono::system_clock> start, end;
	double error;
	
	auto openacc = new float[dimension * dimension];
	auto openaccres = new float[dimension * dimension];
	
	auto cuda = new float[dimension * dimension];
	auto cudares = new float[dimension * dimension];
	if ((algorithms & 0x1) != 0) {
		std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension,
		          &openacc[0]);
		std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension,
		          &openaccres[0]);
		
		start = std::chrono::high_resolution_clock::now();
		openacc_offload(openacc, openaccres, dimension);
		end = std::chrono::high_resolution_clock::now();
		openacc_offload(openaccres, openacc, dimension);
		
		std::chrono::duration<float> openacc_offload_time = end - start;
		
		printf("%f\n", openacc_offload_time.count());
		if (run == 9) {
			printf("------------------------------\n");
			for (int y = 0; y < dimension; y++) {
				for (int x = 0; x < dimension; x++) {
					error = fabs(matrix[y * dimension + x] - openacc[y * dimension + x]);
					
					if (std::isnan(error)) {
						printf("NaN\n");
						continue;
					}
					std::cout << error << std::endl;
				}
			}
		}
	}
	
	
	if ((algorithms & 0x2) != 0) {
		std::copy(matrix, matrix + dimension * dimension,
		          cuda);
		std::copy(identity_matrix, identity_matrix + dimension * dimension,
		          cudares);
		
		start = std::chrono::high_resolution_clock::now();
		cuda_offload(cuda, cudares, dimension);
		end = std::chrono::high_resolution_clock::now();
		cuda_offload(cudares, cuda, dimension);
		
		std::chrono::duration<float> cuda_time = end - start;
		printf("%f\n", cuda_time.count());
		
		if (run == 9) {
			printf("------------------------------\n");
			for (int y = 0; y < dimension; y++) {
				for (int x = 0; x < dimension; x++) {
					error = fabs(matrix[y * dimension + x] - cuda[y * dimension + x]);
					if (std::isnan(error)) {
						printf("NaN\n");
						continue;
					}
					std::cout << error << std::endl;
				}
			}
		}
	}
	
	return 0;
}
