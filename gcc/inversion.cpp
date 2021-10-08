#include <omp.h>
#include <stdio.h>
#include <array>

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
	int i;
#pragma omp target data map(tofrom: matrix[0:dim*dim], iden[0:dim*dim]) map(alloc: i)
	for (i = 0; i < dim; i++) {
//#pragma omp target update to(i)
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
#pragma omp target teams distribute parallel for simd
				for (int x = i; x < dim; x++) { // swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
			}
		}
		
		//normalize
#pragma omp target teams distribute parallel for
		for (int x = i + 1; x < dim + i + 1; x++) {
			float factor = matrix[i * dim + i];
			if (x < dim) {
				matrix[i * dim + x] /= factor;
			} else {
				iden[i * dim + x - dim] /= factor;
			}
		}
		matrix[i * dim + i] = 1;
#pragma omp target update to(matrix[i * dim + i])
		
		//gauss
#pragma omp target teams distribute parallel for
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
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

/*int bound(int coord) {
	if (coord > 2) return 0;
	
	if (coord < 0) return 2;
	
	return coord;
}

float det3x3(float **matrix) {
	float det = 0;
	float diagonal = 0;
	int x = 0, y = 0;
	
	for (int i = 0; i < 3; i++) {
		diagonal = matrix[x][y];
		for (int j = 1; j < 3; j++) {
			x = bound(x++);
			y = bound(y++);
			diagonal *= matrix[x][y];
		}
		x = bound(x++);
		det += diagonal;
	}
	
	for (int i = 0; i < 3; i++) {
		diagonal = matrix[x][y];
		
		for (int j = 1; j < 3; j++) {
			x = bound(x--);
			y = bound(y++);
			diagonal *= matrix[x][y];
		}
		x = bound(x--);
		det -= diagonal;
	}
	return det;
}


void adjugate(float **matrix) {
	std::array<float[3][3], 16> subarrays;
	
	int overx = 0;
	int overy = 0;
	
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			auto arr = subarrays.at(y * dim + x);
			for (int yi = 0; yi < 4; yi++) {
				if (yi == y) {
					overy--;
					continue;
				}
				for (int xi = 0; xi < 4; xi++) {
					if (xi == x) {
						overx--;
						continue;
					}
					arr[xi + overx][yi + overy] = matrix[xi][yi];
				}
				overx++;
			}
			overy++;
		}
	}
	
	
	float det = matrix[0][0] *
	            det3x3(subarrays.at(0)) - matrix[1][0] *
	                                      det3x3(subarrays.at(1)) + matrix[2][0] *
	                                                                det3x3(subarrays.at(2)) - matrix[3][0] *
	                                                                                          det3x3(subarrays.at(3));
	float subdet[16];
	int i = 0;
	for (float **matrix : subarrays) {
		i++;
		matrix[i / 4][i % 4] = (i % 2) ? det / det3x3(matrix) : -det / det3x3(matrix);
	}
}*/
