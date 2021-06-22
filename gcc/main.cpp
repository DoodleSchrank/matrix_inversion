#include <math.h>
#include "inversion.cpp"
#include "../nvcc/inversion.cu"
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

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
	
	
	/*auto eigen = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &eigen[0 * dimension + 0]);
	Eigen::MatrixXf eigenM = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(eigen, dimension, dimension);
	
	auto start = std::chrono::high_resolution_clock::now();
	Eigen::MatrixXf eigenresult = eigenM.inverse();
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> eigen_time = end - start;
	printf("Eigen Time: %04f\n", eigen_time.count());*/
	
	
	/*auto cpu = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &cpu[0 * dimension + 0]);
	auto cpures = new float[dimension * dimension + dimension];
	std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &cpures[0 * dimension + 0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	single_cpu(cpu, cpures, dimension);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> cpu_time = end - start;
	printf("CPU Time: %04f\n", cpu_time.count());*/
	
	
	/*auto openmp = new float[dimension * dimension];
	std::copy(&matrix[0], &matrix[0] + dimension * dimension, &openmp[0]);
	auto openmpres = new float[dimension * dimension];
	std::copy(&identity_matrix[0], &identity_matrix[0] + dimension * dimension, &openmpres[0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	openmp_offload(openmp, openmpres, dimension);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openmp_offload_time = end - start;
	printf("%04f\n", openmp_offload_time.count());*/
	
	
	
	auto openacc = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &openacc[0 * dimension + 0]);
	auto openaccres = new float[dimension * dimension + dimension];
	std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &openaccres[0 * dimension + 0]);
	
	auto start = std::chrono::high_resolution_clock::now();
	openacc_offload(openacc, openaccres, dimension);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openacc_offload_time = end - start;
	printf("%04f\n", openacc_offload_time.count());
	//printf("OpenACC offload Time: %04f\n", openacc_offload_time.count());
	
	
	
	/*auto cuda = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &cuda[0 * dimension + 0]);
	auto cudares = new float[dimension * dimension + dimension];
	std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &cudares[0 * dimension + 0]);
	
	start = std::chrono::high_resolution_clock::now();
	cuda(cuda, cudares, dimension);
	end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> cuda_time = end - start;
	printf("CUDA Time: %04f\n", cuda_time.count());*/
	
	
	/*double threshold = 0.00001;
	int errc = 0;
	double minerr = 1000000.;
	double maxerr = threshold;
	double maxerrval;
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			double error = fabs(openmpres[i * dimension + j] - eigenresult(i, j));
			//float error = abs(cpures[i * dimension + j] - openmpres[i * dimension + j]);
			if (error > threshold || std::isnan(error)) {
				errc++;
				//std::cout << "Error at " << i << " " << j << std::endl;
				//std::cout << eigenresult(i, j) << " vs " << openmpres[i * dimension + j] << std::endl;
			}
			if (maxerr < error) {
				maxerr = error;
				maxerrval = openmpres[i * dimension + j];
			}
			if (minerr > error) minerr = error;
		}
	}
	printf("errc: %d\n", errc);
	printf("maxerrval: %04f maxerr: %04f  minerr: %04f\n", maxerrval, maxerr, minerr);*/
	
	
	/*for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			printf("%04f ", openmpres[i * dimension + j]);
		}
		printf("  \t");
		for (int j = 0; j < dimension; j++) {
			printf("%04f ", eigenresult(i, j));
		}
		printf("  \t");
		for (int j = 0; j < dimension; j++) {
			printf("%04f ", cpures[i * dimension + j]);
		}
		printf("\n");
	}*/

return 0;
}
