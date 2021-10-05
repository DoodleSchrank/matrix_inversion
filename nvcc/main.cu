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
	double threshold = 0.00001;
	int errc;
	double minerr = 1000000.;
	double maxerr = threshold;
	
	
	if ((algorithms & 0x1) != 0) {
		auto openacc = new float[dimension * dimension + dimension];
		std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension,
		          &openacc[0]);
		auto openaccres = new float[dimension * dimension + dimension];
		std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension,
		          &openaccres[0]);
		
		start = std::chrono::high_resolution_clock::now();
		openacc_offload(openacc, openaccres, dimension);
		end = std::chrono::high_resolution_clock::now();
		openacc_offload(openaccres, openacc, dimension);
		
		std::chrono::duration<float> openacc_offload_time = end - start;
		
		errc = 0;
		minerr = 10000000000000000.;
		maxerr = threshold;
		std::string msg;
		for (int y = 0; y < dimension; y++) {
			for (int x = 0; x < dimension; x++) {
				error = fabs(matrix[y * dimension + x] - openacc[y * dimension + x]);
				
				if (error > threshold || std::isnan(error)) {
					errc++;
					if (maxerr < error) {
						maxerr = error / matrix[y * dimension + x];
					}
					if (minerr > error) minerr = error / matrix[y * dimension + x];
				}
				
			}
		}
		printf("OpenACC #err: %d - minerr:%04f - maxerr:%04f\n", errc, minerr, maxerr);
		printf("OpenACC: %04f\n", openacc_offload_time.count());
	}
	
	
	if ((algorithms & 0x2) != 0) {
		auto cuda = new float[dimension * dimension + dimension];
		std::copy(&matrix[0 * dimension + 0], &matrix[0 * dimension + 0] + dimension * dimension,
		          &cuda[0 * dimension + 0]);
		auto cudares = new float[dimension * dimension + dimension];
		std::copy(&identity_matrix[0 * dimension + 0], &identity_matrix[0 * dimension + 0] + dimension * dimension,
		          &cudares[0 * dimension + 0]);
		
		start = std::chrono::high_resolution_clock::now();
		cuda_offload(cuda, cudares, dimension);
		end = std::chrono::high_resolution_clock::now();
		cuda_offload(cudares, cuda, dimension);
		
		std::chrono::duration<float> cuda_time = end - start;
		printf("CUDA: %04f\n", cuda_time.count());
		
		errc = 0;
		minerr = 10000000000000000.;
		maxerr = threshold;
		for (int y = 0; y < dimension; y++) {
			for (int x = 0; x < dimension; x++) {
				error = fabs(matrix[y * dimension + x] - cuda[y * dimension + x]);
				
				if (error > threshold || std::isnan(error)) {
					errc++;
					if (maxerr < error) {
						maxerr = error / matrix[y * dimension + x];
					}
					if (minerr > error) minerr = error / matrix[y * dimension + x];
				}
				
			}
		}
		printf("CUDA #err: %d - minerr:%04f - maxerr:%04f\n", errc, minerr, maxerr);
	}
	
	/*for (int y = 0; y < dimension; y++) {
		for (int x = 0; x < dimension; x++) {
			error = fabs(cuda[y * dimension + x] - openacc[y * dimension + x]);

			if (error > threshold || std::isnan(error)) {
				errc++;
				if (maxerr < error) {
					maxerr = error / matrix[y * dimension + x];
				}
				if (minerr > error) minerr = error / matrix[y * dimension + x];
			}
			
		}
	}
	printf("CUDA vs OpenACC #err: %d - minerr:%04f - maxerr:%04f\n", errc, minerr, maxerr);*/
	
	
	
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
