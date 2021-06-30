#include <omp.h>
#include <stdio.h>

void single_cpu(float *matrix, float *iden, int dim) {
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
				for (int x = i; x < dim; x++) { // swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
				
			}
		}
		
		//normalize
		float factor = matrix[i * dim + i];
		for (int x = 0; x < 2 * dim; x++) {
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else
				iden[i * dim + x - dim] /= factor;
		}
		
		//gauss
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y * dim + x] -= matrix[i * dim + x] * factor;
					else
						iden[y * dim + x - dim] -= iden[i * dim + x - dim] * factor;
				}
			}
		}
	}
}


void openmp_offload(float *matrix, float *iden, int dim) {
	float factor;
#pragma omp target data map(tofrom: matrix[0:dim*dim], iden[0:dim*dim]) map(alloc: factor)
#pragma omp target
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
#pragma omp teams distribute parallel for simd
				for (int x = i; x < dim; x++) { // swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
			}
		}
		
		
		//normalize
//#pragma omp target update from(matrix[i * dim + i])
		factor = matrix[i * dim + i];
#pragma omp teams distribute shared(factor)
		for (int x = i; x < dim + i + 1; x++) {
			if (x >= 2 * dim)
				printf("%d", x);
			//factor = matrix[i * dim + i];
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else
				iden[i * dim + x - dim] /= factor;
		}
		
		//gauss
#pragma omp teams distribute parallel for private(factor)
		for (int y = 0; y < dim; y++) {
			factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
#pragma omp simd
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim)
						matrix[y * dim + x] -= matrix[i * dim + x] * factor;
					else
						iden[y * dim + x - dim] -= iden[i * dim + x - dim] * factor;
				}
			}
		}
	}
//#pragma omp target exit data map(from: matrix[0:dim*dim], iden[0:dim*dim])
}
