#include <math.h>
#include "inversion.cpp"
//#include "../nvcc/inversion.cu"
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>

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
	std::cout << std::fixed << std::setprecision(10);
	
	
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
	
	
	if ((algorithms & 0x1) != 0) {
		auto eigen = new float[dimension * dimension];
		std::copy(matrix, matrix + dimension * dimension, eigen);
		Eigen::MatrixXf
		eigenM = Eigen::Map < Eigen::Matrix < float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
				>> (eigen, dimension, dimension);
		
		start = std::chrono::high_resolution_clock::now();
		Eigen::MatrixXf eigenresult = eigenM.inverse();
		end = std::chrono::high_resolution_clock::now();
		eigenresult = eigenresult.inverse();
		
		std::chrono::duration<float> eigen_time = end - start;
		printf("%f\n", eigen_time.count());
		
		if (run == 9) {
			printf("------------------------------\n");
			for (int y = 0; y < dimension; y++) {
				for (int x = 0; x < dimension; x++) {
					error = fabs(eigenresult(y, x) - eigenM(y, x));
					if (std::isnan(error)) {
						printf("NaN\n");
						return 0;
					}
					std::cout << error << std::endl;
				}
			}
		}
	}
	
	if ((algorithms & 0x2) != 0) {
		auto cpu = new float[dimension * dimension + dimension];
		std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension,
		          &cpu[0 * dimension + 0]);
		auto cpures = new float[dimension * dimension + dimension];
		std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension,
		          &cpures[0 * dimension + 0]);
		
		start = std::chrono::high_resolution_clock::now();
		single_cpu(cpu, cpures, dimension);
		end = std::chrono::high_resolution_clock::now();
		single_cpu(cpures, cpu, dimension);
		
		std::chrono::duration<float> cpu_time = end - start;
		printf("%f\n", cpu_time.count());
		
		if (run == 9) {
			printf("------------------------------\n");
			for (int y = 0; y < dimension; y++) {
				for (int x = 0; x < dimension; x++) {
					error = fabs(matrix[y * dimension + x] - cpu[y * dimension + x]);
					if (std::isnan(error)) {
						printf("NaN\n");
						return 0;
					}
					std::cout << error << std::endl;
				}
			}
		}
	}
	
	if ((algorithms & 0x4) != 0) {
		auto openmp = new float[dimension * dimension];
		std::copy(&matrix[0], &matrix[0] + dimension * dimension, &openmp[0]);
		auto openmpres = new float[dimension * dimension];
		std::copy(&identity_matrix[0], &identity_matrix[0] + dimension * dimension, &openmpres[0]);
		
		start = std::chrono::high_resolution_clock::now();
		openmp_offload(openmp, openmpres, dimension);
		end = std::chrono::high_resolution_clock::now();
		openmp_offload(openmpres, openmp, dimension);
		
		std::chrono::duration<float> openmp_offload_time = end - start;
		printf("%f\n", openmp_offload_time.count());
		
		if (run == 9) {
			printf("------------------------------\n");
			for (int y = 0; y < dimension; y++) {
				for (int x = 0; x < dimension; x++) {
					error = fabs(matrix[y * dimension + x] - openmp[y * dimension + x]);
					if (std::isnan(error)) {
						printf("NaN\n");
						return 0;
					}
					std::cout << error << std::endl;
				}
			}
		}
	}
	
	return 0;
}
