#include <cmath>
#include <iostream>
#include <fstream>
#include "inversion.cu"
//#include "matrixInversion_gpu.cu"
#include <cuda_runtime.h>
//#include <Eigen/Core>
//#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <map>
#include <fmt/format.h>
#include <sstream>

const int dimension = 1024;
float matrix[dimension][dimension];
float identity_matrix[dimension][dimension];


void matrix_read() {
	int row = 0;
	
	std::ifstream infile(("randomMatrix_" + std::to_string(dimension) + ".txt").c_str());
	for (std::string line; std::getline(infile, line);) {
		std::istringstream inputline(line);
		for (int i = 0; i < dimension; i++) {
			inputline >> matrix[row][i];
		}
		row++;
	}
}

int main() {
	matrix_read();
	
	//fill identity matrix
#pragma omp parallel for collapse(2)
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			if (i == j) {
				identity_matrix[i][i] = 1;
			} else {
				identity_matrix[i][j] = 0;
			}
		}
	}
	
	
	
	auto openmp = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &openmp[0][0]);
	auto openmpres = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &openmpres[0][0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	openmp_offload(openmp, openmpres);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openmp_offload_time = end - start;
	std::cout << "OpenMP Offload Time - inverse:\n" << openmp_offload_time.count() << "s" << std::endl;
	
	
	
	auto cuda = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &cuda[0][0]);
	auto cudares = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &cudares[0][0]);
	
	start = std::chrono::high_resolution_clock::now();
	cuda_offload(cuda, cudares);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> cuda_time = end - start;
	std::cout << "CUDA Offload Time - inverse:\n" << cuda_time.count() << "s" << std::endl;
	
	
	float threshold = 0.00001;
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
	std::cout << "maxerrval: " << maxerrval << " maxerr: " << maxerr << "  minerr: " << minerr << std::endl;
	
	
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
