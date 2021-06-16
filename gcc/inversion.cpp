#include <omp.h>

#define minvalue 0.0000

template<int dim>
void single_cpu(float matrix[dim][dim], float iden[dim][dim]) {
	float factor;
	for (int i = 0; i < dim; i++) {
		if (matrix[i][i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j][i] != 0) {
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i][x] += matrix[j][x];
						iden[i][x] = iden[j][x];
					}
					break;
				}
			}
		}
		
		//normalize
		factor = matrix[i][i];
		for (int x = 0; x < 2 * dim; x++) {
			if (x < dim)
				matrix[i][x] /= factor;
			else
				iden[i][x - dim] /= factor;
		}
		
		//gauss
		for (int y = 0; y < dim; y++) {
			factor = matrix[y][i];
			if (y != i && factor != 0.0f) {
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y][x] -= matrix[i][x] * factor;
					else
						iden[y][x - dim] -= iden[i][x - dim] * factor;
				}
			}
		}
	}
}


template<int dim>
void openmp_offload(float matrix[dim][dim], float iden[dim][dim]) {
#pragma omp target enter data map(to: matrix[0:dim][0:dim], iden[0:dim][0:dim])
	for (int i = 0; i < dim; i++) {
		if (matrix[i][i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j][i] != 0) {
#pragma omp target teams distribute parallel for simd
					for (int x = i; x < dim; x++) { // swap lines
						matrix[i][x] += matrix[j][x];
						iden[i][x] = iden[j][x];
					}
					break;
				}
			}
		}
		
		
		//normalize
#pragma omp target teams distribute
		for (int x = i; x < 2 * dim; x++) {
			float factor = matrix[i][i];
			if (x < dim)
				matrix[i][x] /= factor;
			else
				iden[i][x - dim] /= factor;
		}
		
		//gauss
#pragma omp target teams distribute parallel for
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y][i];
			if (y != i && factor != 0.0f) {
#pragma omp simd
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y][x] -= matrix[i][x] * factor;
					else
						iden[y][x - dim] -= iden[i][x - dim] * factor;
				}
			}
		}
	}
#pragma omp target exit data map(from: iden[0:dim][0:dim])
}
