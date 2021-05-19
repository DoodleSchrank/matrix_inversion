#include <cmath>
#include <iostream>
#include <fstream>
#include "inversion.cpp"
//#include "matrixInversion_gpu.cu"
//#include <cuda_runtime.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <map>
#include <fmt/format.h>
const int dimension = 1000;
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
	
	/*auto eigen = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &eigen[0][0]);
	auto eigenM = new Eigen::Map<Eigen::Matrix<float, dimension, dimension, Eigen::RowMajor>>(&eigen[0][0], dimension, dimension);
	
	auto start = std::chrono::high_resolution_clock::now();
	auto eigenresult = eigenM->inverse();
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> eigen_time = end - start;
	std::cout << "Eigen Time - inverse:\n" << eigen_time.count() << "ms" << std::endl;*/
	
	
	auto cpu = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &cpu[0][0]);
	auto cpures = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &cpures[0][0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	single_cpu(cpu, cpures);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> cpu_time = end - start;
	std::cout << "CPU Time - inverse:\n" << cpu_time.count() << "ms" << std::endl;
	
	
	auto openmp = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &openmp[0][0]);
	auto openmpres = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &openmpres[0][0]);
#pragma omp target enter data map(to: openmp[0:dimension][0:dimension], openmpres[0:dimension][0:dimension])
	
	start = std::chrono::high_resolution_clock::now();
	openmp_offload(openmp, openmpres);
	end = std::chrono::high_resolution_clock::now();

#pragma omp target exit data map(from: openmpres[0:dimension][0:dimension])
	std::chrono::duration<float> openmp_offload_time = end - start;
	std::cout << "OpenMP Offload Time - inverse:\n" << openmp_offload_time.count() << "ms" << std::endl;
	
	
	/*start = std::chrono::high_resolution_clock::now();
	openacc_offload(matrix, identity_matrix, dimension);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> openacc_offload_time = end - start;
	std::cout << "OpenACC Offload Time - inverse:\n" << openacc_offload_time.count() << "s" << std::endl;*/
	
	/*start = std::chrono::high_resolution_clock::now();
	cuda(&matrix, dimension);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> cuda_time = end - start;
	std::cout << "CUDA Time - inverse:\n" << cuda_time << "ms" << std::endl;*/
	
	
	float threshold = 0.00001;
	int errc = 0;
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			if (abs(cpures[i][j] - openmpres[i][j]) > threshold) {
				errc++;
				//std::cout << "Error at " << i << " " << j << std::endl;
				//std::cout << eigenresult(i, j) << " vs " << openmpres[i][j] << std::endl;
			}
		}
	}
	std::cout << "errc: " << errc << std::endl;
	std::cout << cpures[1][1] << " vs " << openmpres[1][1] << std::endl;
	return 0;
}
