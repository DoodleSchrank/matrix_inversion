#include <math.h>
#include "inversion.cpp"
#include "../nvcc/inversion.cu"
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <map>

const int dimension = 5000;
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
	
	auto eigen = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &eigen[0][0]);
	Eigen::MatrixXf eigenM = Eigen::Map<Eigen::Matrix<float, dimension, dimension, Eigen::RowMajor>>((float *) eigen, dimension, dimension);
	
	auto start = std::chrono::high_resolution_clock::now();
	Eigen::MatrixXf eigenresult = eigenM.inverse();
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> eigen_time = end - start;
	std::cout << "Eigen Time - inverse:\n" << eigen_time.count() << "s" << std::endl;
	
	
	/*auto cpu = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &cpu[0][0]);
	auto cpures = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &cpures[0][0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	single_cpu(cpu, cpures);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> cpu_time = end - start;
	std::cout << "CPU Time - inverse:\n" << cpu_time.count() << "s" << std::endl;*/
	
	
	/*auto openmp = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &openmp[0][0]);
	auto openmpres = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &openmpres[0][0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	openmp_offload(openmp, openmpres);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openmp_offload_time = end - start;
	std::cout << "OpenMP Offload Time - inverse:\n" << openmp_offload_time.count() << "s" << std::endl;
	
	
	
	auto openacc = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &openacc[0][0]);
	auto openaccres = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &openaccres[0][0]);
	
	start = std::chrono::high_resolution_clock::now();
	openacc_offload(openacc, openaccres);
	end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openacc_offload_time = end - start;
	std::cout << "OpenACC Offload Time - inverse:\n" << openacc_offload_time.count() << "s" << std::endl;
	
	
	
	auto cuda = new float[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &cuda[0][0]);
	auto cudares = new float[dimension][dimension];
	std::copy(&identity_matrix[0][0], &identity_matrix[0][0] + dimension * dimension, &cudares[0][0]);
	
	start = std::chrono::high_resolution_clock::now();
	cuda(cuda, cudares);
	end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> cuda_time = end - start;
	std::cout << "CUDA Time - inverse:\n" << cuda_time << "ms" << std::endl;*/
	
	
	/*float threshold = 0.00001;
	int errc = 0;
	float minerr = 1000000.;
	float maxerr = threshold;
	float maxerrval;
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			float error = fabs(openmpres[i][j] - openaccres[i][j]);
			//float error = abs(cpures[i][j] - openmpres[i][j]);
			if (error > threshold || std::isnan(error)) {
				errc++;
				//std::cout << "Error at " << i << " " << j << std::endl;
				//std::cout << eigenresult(i, j) << " vs " << openmpres[i][j] << std::endl;
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
			std::cout << openmpres[i][j] << " ";
		}
		for (int j = 0; j < dimension; j++) {
			std::cout << openaccres[i][j] << " ";
		}
		std::cout << std::endl;
	}*/

return 0;
}
