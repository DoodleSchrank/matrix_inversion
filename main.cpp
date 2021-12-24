#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string.h>

#if (defined __GNUC__ || defined __clang__) && !defined(__NVCOMPILER)
	#include "implementations/openmp-cpu.cpp"
	#include "implementations/openmp-offload.cpp"
	#include "implementations/opencl.cpp"
	#include "implementations/openacc.cpp"
	#include <Eigen/Core>
	#include <Eigen/Dense>
#endif

#if defined __NVCOMPILER
	#include "implementations/cublas.cpp"
	#include "implementations/cuda.cpp"
	#include "implementations/openacc.cpp"
	#include "implementations/openmp-cpu.cpp"

	#if __CUDA_ARCH__ > 700
		#include "implementations/openmp-offload.cpp"
	#endif

#endif

#ifdef dbl
	using scalar = double;
#else
	using scalar = float;
#endif

#if (defined __GNUC__ || defined __clang__) && !defined(__NVCOMPILER)
typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
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

#if (defined __GNUC__ || defined __clang__) && !defined(__NVCOMPILER)
	MatrixXs eigenM;
	MatrixXs eigenresult;
	if (!strcmp(algorithm, "eigen")) {
		eigenM = Eigen::Map<MatrixXs>(calc_matrix, dimension, dimension);
		start = std::chrono::high_resolution_clock::now();
		eigenresult = eigenM.inverse();
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());

		start = std::chrono::high_resolution_clock::now();
		eigenresult = eigenresult.inverse();
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());
	}
#endif

	if (!strcmp(algorithm, "openmp-cpu")) {
		start = std::chrono::high_resolution_clock::now();
		openmp_cpu(calc_matrix, calc_identity, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());

		start = std::chrono::high_resolution_clock::now();
		openmp_cpu(calc_identity, calc_matrix, dimension);
		end = std::chrono::high_resolution_clock::now();
		measurement = end - start;
		printf("%f\n", measurement.count());
	}

#if (defined __GNUC__ || defined __clang__) && !defined(__NVCOMPILER)
	if (!strcmp(algorithm, "opencl")) {
		start = std::chrono::high_resolution_clock::now();
		opencl_offload(calc_matrix, calc_identity, dimension);
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

#if (defined __GNUC__ || defined __clang__) || __CUDA_ARCH__ > 700
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

#ifdef __NVCOMPILER
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
#endif
#if defined __GNUC__ || __NVCOMPILER && !defined(__NVCOMPILER)
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
#endif

	if (run == 4) {
		printf("------------------------------\n");
		for (int y = 0; y < dimension; y++) {
			for (int x = 0; x < dimension; x++) {
				if (strcmp(algorithm, "eigen")) {
					error = fabs(matrix[y * dimension + x] - calc_matrix[y * dimension + x]);
				}
				#if (defined __GNUC__ || defined __clang__) && !defined(__NVCOMPILER)
				else {
					error = fabs(eigenresult(y, x) - eigenM(y, x));
				}
				#endif
				
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
