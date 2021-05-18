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

const int dimension = 1000;
float matrix[dimension][dimension];
float identity_matrix[dimension][dimension];

void out_txt(const std::string& filename, int j_begin) {
	std::ofstream ofile;
	ofile.open(filename, std::ios::out | std::ios::app);
	for (auto & i : matrix) {
		for (int j = j_begin; j < dimension; j++)
			ofile << i[j] << ",";
		ofile << "\n";
	}
	ofile.close();                //close output file
}

void matrix_read() {
	int row = 0;
	
	std::ifstream infile("randomMatrix_" + std::to_string(dimension) + ".txt");
	
	for (std::string line; std::getline(infile, line);) {
		std::istringstream inputline(line);
		for (int i = 0; i < dimension; i++) {
			inputline >> matrix[row][i];
		}
		row++;
	}
	
}

int main() {
	std::cout << "INVERSE OF NON-SINGULAR MATRIX BY GAUSS-JORDAN ELIMINATION METHOD" << std::endl;
	
	matrix_read();

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
	
	float eigen[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &eigen[0][0]);
	auto eigenM = Eigen::Map < Eigen::Matrix < float, dimension, dimension, Eigen::RowMajor>>(&eigen[0][0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	auto eigenresult = eigenM.inverse();
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<double> eigen_time = end - start;
	std::cout << "Eigen Time - inverse:\n" << eigen_time.count() << "ms" << std::endl;
	
	
	/*start = std::chrono::high_resolution_clock::now();
	single_cpu(matrix, identity_matrix, dimension);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> cpu_time = end - start;
	std::cout << "CPU Time - inverse:\n" << cpu_time.count() << "ms" << std::endl;*/
	
	
	float openmp[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &openmp[0][0]);
	float openmpres[dimension][dimension];
	std::copy(&matrix[0][0], &matrix[0][0] + dimension * dimension, &openmpres[0][0]);
	#pragma omp target enter data map(to: openmp[0:dimension][0:dimension], openmpres[0:dimension][0:dimension])
	
	start = std::chrono::high_resolution_clock::now();
	openmp_offload(openmp, openmpres, dimension);
	end = std::chrono::high_resolution_clock::now();

#pragma omp target exit data map(from: openmpres[dimension][dimension])
	std::chrono::duration<double> openmp_offload_time = end - start;
	std::cout << "OpenMP Offload Time - inverse:\n" << openmp_offload_time.count() << "ms" << std::endl;
	
	
	/*start = std::chrono::high_resolution_clock::now();
	openacc_offload(matrix, identity_matrix, dimension);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> openacc_offload_time = end - start;
	std::cout << "OpenACC Offload Time - inverse:\n" << openacc_offload_time.count() << "s" << std::endl;*/
	
	/*start = std::chrono::high_resolution_clock::now();
	cuda(&matrix, dimension);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> cuda_time = end - start;
	std::cout << "CUDA Time - inverse:\n" << cuda_time << "ms" << std::endl;*/
	
	/*for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			std::cout << eigenresult(i, j) - openmpres[i][j] << " ";
		}
		std::cout << std::endl;
	}*/
	
	std::cout << eigenresult(0) << std::endl;
	
	return 0;
}
