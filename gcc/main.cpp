#include <math.h>
#include "inversion.cpp"
//#include "../nvcc/inversion.cu"
#include <iostream>
#include <fstream>
#include <sstream>
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
	int algorithms = std::stoi(argv[2]);
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
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	double error;
	double threshold = 0.00001;
	int errc;
	double minerr = 1000000.;
	double maxerr = threshold;
	
	if((algorithms & 0x1) != 0) {
		auto eigen = new float[dimension * dimension + dimension];
		std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &eigen[0 * dimension + 0]);
		Eigen::MatrixXf eigenM = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(eigen, dimension, dimension);
		
		start = std::chrono::high_resolution_clock::now();
		Eigen::MatrixXf eigenresult = eigenM.inverse();
		end = std::chrono::high_resolution_clock::now();
		eigenresult = eigenresult.inverse();
		
		std::chrono::duration<float> eigen_time = end - start;
		printf("Eigen Time: %04f\n", eigen_time.count());
		
		errc = 0;
		minerr = 1000000.;
		maxerr = threshold;
		for (int y = 0; y < dimension; y++) {
			for(int x = 0; x < dimension; x++) {
				error = fabs(matrix[y * dimension + x] - eigenresult(y, x));
				if(std::isnan(error)) {
					printf("NaN\n");
					return 0;
				}
				if (error > threshold) {
					errc++;
					if (maxerr < error / matrix[y * dimension + x]) {
						maxerr = error / matrix[y * dimension + x];
					}
					if (minerr > error / matrix[y * dimension + x]) minerr = error / matrix[y * dimension + x];
				}
				
			}
		}
		printf("Eigen #err: %d - minerr:%04f - maxerr:%04f\n", errc, minerr, maxerr);
	}
	
	if((algorithms & 0x2) != 0) {
		auto cpu = new float[dimension * dimension + dimension];
		std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &cpu[0 * dimension + 0]);
		auto cpures = new float[dimension * dimension + dimension];
		std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &cpures[0 * dimension + 0]);
		
		start = std::chrono::high_resolution_clock::now();
		single_cpu(cpu, cpures, dimension);
		end = std::chrono::high_resolution_clock::now();
		single_cpu(cpures, cpu, dimension);
		
		std::chrono::duration<float> cpu_time = end - start;
		printf("CPU Time: %04f\n", cpu_time.count());
		
		errc = 0;
		minerr = 1000000.;
		maxerr = threshold;
		for (int y = 0; y < dimension; y++) {
			for(int x = 0; x < dimension; x++) {
				error = fabs(matrix[y * dimension + x] - cpu[y * dimension + x]);
				if(std::isnan(error)) {
					printf("NaN\n");
					return 0;
				}
				if (error > threshold) {
					errc++;
					if (maxerr < error) {
						maxerr = error / matrix[y * dimension + x];
					}
					if (minerr > error) minerr = error / matrix[y * dimension + x];
				}
				
			}
		}
		printf("CPU #err: %d - minerr:%04f - maxerr:%04f\n", errc, minerr, maxerr);
	}
	
	if((algorithms & 0x4) != 0) {
		auto openmp = new float[dimension * dimension];
		std::copy(&matrix[0], &matrix[0] + dimension * dimension, &openmp[0]);
		auto openmpres = new float[dimension * dimension];
		std::copy(&identity_matrix[0], &identity_matrix[0] + dimension * dimension, &openmpres[0]);
		
		start = std::chrono::high_resolution_clock::now();
		openmp_offload(openmp, openmpres, dimension);
		end = std::chrono::high_resolution_clock::now();
		openmp_offload(openmpres, openmp, dimension);
		
		std::chrono::duration<float> openmp_offload_time = end - start;
		printf("OpenMP: %04f\n", openmp_offload_time.count());
		
		errc = 0;
		minerr = 1000000.;
		maxerr = threshold;
		for (int y = 0; y < dimension; y++) {
			for(int x = 0; x < dimension; x++) {
				error = fabs(matrix[y * dimension + x] - openmp[y * dimension + x]);
				if(std::isnan(error)) {
					printf("NaN\n");
					return 0;
				}
				if (error > threshold) {
					errc++;
					if (maxerr < error) maxerr = error / matrix[y * dimension + x];
					else if (minerr > error) minerr = error / matrix[y * dimension + x];
				}
				
			}
		}
		printf("OpenMP #err: %d - minerr:%04f - maxerr:%04f\n", errc, minerr, maxerr);
		/*for (int i = 0; i < dimension; i++) {
			for (int j = 0; j < dimension; j++) {
				printf("%04f ", matrix[i * dimension + j]);
			}
			printf("  \t");
			for (int j = 0; j < dimension; j++) {
				printf("%04f ", openmp[i * dimension + j]);
			}
			printf("  \n");
		}*/
	}
	
	
	
	/*auto openacc = new float[dimension * dimension + dimension];
	std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension, &openacc[0 * dimension + 0]);
	auto openaccres = new float[dimension * dimension + dimension];
	std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension, &openaccres[0 * dimension + 0]);
	
	start = std::chrono::high_resolution_clock::now();
	openacc_offload(openacc, openaccres, dimension);
	end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<float> openacc_offload_time = end - start;
	printf("%04f\n", openacc_offload_time.count());
	
	
	
	auto cuda = new float[dimension * dimension + dimension];
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
