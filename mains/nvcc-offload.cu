#include <algorithm>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>
#include "../implementations/cublas.cu"
#include "../implementations/cuda.cu"
#include "../implementations/openacc.cpp"

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif


void matrix_read(char *file, int dim, scalar *matrix) {
	int row = 0;

	std::ifstream infile(file);
	for (std::string line; std::getline(infile, line);) {
		std::istringstream inputline(line);
		for (int i = 0; i < dim; i++) {
			inputline >> matrix[row * dim + i];
		}
		row++;
	}
}

int main(int argc, char *argv[]) {
	std::cout << std::fixed << std::setprecision(10);

	char *file = argv[1];
	int dimension = std::stoi(argv[2]);
	char *algorithm = argv[3];
	int run = std::stoi(argv[4]);
	auto matrix = new scalar[dimension * dimension];
	auto identity_matrix = new scalar[dimension * dimension];

	matrix_read(file, dimension, static_cast<scalar *>(matrix));

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

	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<scalar> measurement;
	double error;

	scalar *calc_matrix = new scalar[dimension * dimension];
	scalar *calc_identity = new scalar[dimension * dimension];
#pragma omp parallel for collapse(1)
	for (int y = 0; y < dimension; y++) {
		for (int x = 0; x < dimension; x++) {
			calc_matrix[y * dimension + x] = matrix[y * dimension + x];
			calc_identity[y * dimension + x] = identity_matrix[y * dimension + x];
		}
	}

#if __CUDA_ARCH__ > 700
	if (!strcmp(algorithm, "openmp-offload")) {
		start = std::chrono::high_resolution_clock::now();
		openmp_offload(calc_matrix, calc_identity, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());

		start = std::chrono::high_resolution_clock::now();
		openmp_offload(calc_identity, calc_matrix, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());
	}
#endif
	
	if (!strcmp(algorithm, "cuda")) {
		start = std::chrono::high_resolution_clock::now();
		cuda_offload(calc_matrix, calc_identity, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());

		start = std::chrono::high_resolution_clock::now();
		cuda_offload(calc_identity, calc_matrix, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());
	}

	if (!strcmp(algorithm, "cublas")) {
		start = std::chrono::high_resolution_clock::now();
		cublas_offload(calc_matrix, calc_identity, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());

		start = std::chrono::high_resolution_clock::now();
		cuda_offload(calc_identity, calc_matrix, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());
	}
	if (!strcmp(algorithm, "openacc")) {
		start = std::chrono::high_resolution_clock::now();
		openacc_offload(calc_matrix, calc_identity, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());

		start = std::chrono::high_resolution_clock::now();
		openacc_offload(calc_identity, calc_matrix, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());
	}

	if (run == 4) {
		printf("------------------------------\n");
		for (int y = 0; y < dimension; y++) {
			for (int x = 0; x < dimension; x++) {
				error = matrix[y * dimension + x] - calc_matrix[y * dimension + x];
				if (std::isnan(error)) {
					printf("NaN\n");
					return 0;
				}
				std::cout << error << std::endl;
			}
		}
	}
	return 0;
}
