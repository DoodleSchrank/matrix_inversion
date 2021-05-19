#include <stdio.h>
#include <iostream>
#include <omp.h>

#define minvalue 0.0000

template<int dim>
void single_cpu(float matrix[dim][dim], float iden[dim][dim]) {
	for (int i = 0; i < dim; i++) {
		if (matrix[i][i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j][i] != 0) {
					int swap;
					for (int y = i; y < dim; y++) { // swap lines
						swap = matrix[i][y];
						matrix[i][y] = matrix[j][y];
						matrix[j][y] = swap;
						swap = iden[i][y];
						iden[i][y] = iden[j][y];
						iden[j][y] = swap;
					}
					break;
				}
			}
		}
		
		//normalize
		float factor = matrix[i][i];
		for (int x = 0; x < dim; x++) {
			iden[i][x] /= factor;
			matrix[i][x] /= factor;
		}
		for (int y = 0; y < dim; y++) {
			factor = matrix[y][i];
			if (y != i && factor != 0.0f) {
				for (int x = 0; x < dim; x++) {
					matrix[y][x] -= matrix[i][x] * factor;
					iden[y][x] -= iden[i][x] * factor;
				}
			}
		}
	}
}


template<int dim>
void openmp_offload(float matrix[dim][dim], float iden[dim][dim]) {
//#pragma omp target
	{
		for (int i = 0; i < dim; i++) {
			if (matrix[i][i] == 0) { // swap lines if 0
				for (int j = i + 1; j < dim; j++) { // find new line
					if (matrix[j][i] != 0) {
						int swap;
#pragma omp target teams distribute parallel for private(swap)
						for (int y = i; y < dim; y++) { // swap lines
							swap = matrix[i][y];
							matrix[i][y] = matrix[j][y];
							matrix[j][y] = swap;
							swap = iden[i][y];
							iden[i][y] = iden[j][y];
							iden[j][y] = swap;
						}
						break;
					}
				}
			}
			
			//normalize
			float factor = matrix[i][i];
#pragma omp target teams distribute parallel for
			for (int x = 0; x < dim; x++) {
				iden[i][x] /= factor;
				matrix[i][x] /= factor;
			}
			
			//gauss
#pragma omp target teams distribute parallel for shared(matrix, iden, i)
			for (int y = 0; y < dim; y++) {
				factor = matrix[y][i];
				if (y != i && factor != 0.0f) {
					for (int x = 0; x < dim; x++) {
						matrix[y][x] -= matrix[i][x] * factor;
						iden[y][x] -= iden[i][x] * factor;
					}
				}
			}
		}
	}
}

/*template<size_t rows, size_t columns>
void openacc_offload(float (&matrix)[rows][columns], float (&iden)[irows][icolumns], int dimension) {
	float temporary, r;
	int j, i, k, temp;
	for (j = 0; j < dimension; j++) {
		temp = j;
		
		finding maximum jth column element in last (dimension-j) rows
		
		for (i = j + 1; i < dimension; i++)
			if (matrix[i][j] > matrix[temp][j])
				temp = i;
		
		if (fabs(matrix[temp][j]) < minvalue) {
			printf("\n Elements are too small to deal with !!!");
			break;
		}
		
		swapping row which has maximum jth column element
		
		if (temp != j)
			for (k = 0; k < 2 * dimension; k++) {
				temporary = matrix[j][k];
				matrix[j][k] = matrix[temp][k];
				matrix[temp][k] = temporary;
			}
		
		performing row operations to form required identity matrix out of the input matrix
		
		for (i = 0; i < dimension; i++)
			if (i != j) {
				r = matrix[i][j];
				for (k = 0; k < 2 * dimension; k++)
					matrix[i][k] -= (matrix[j][k] / matrix[j][j]) * r;
			} else {
				r = matrix[i][j];
				for (k = 0; k < 2 * dimension; k++)
					matrix[i][k] /= r;
			}
		
	}
}*/
