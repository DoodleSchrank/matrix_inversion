#include <stdio.h>
#include <omp.h>

#define minvalue 0.0000

/*template<size_t rows, size_t columns>
void single_cpu(float (&matrix)[rows][columns], float (&iden)[irows][icolumns], int dimension) {
	float temporary, r;
	int j, i, k, temp;
	
	for (j = 0; j < dimension; j++) {
		temp = j;
		
		finding maximum jth column element in last (dimension-j) rows
		
		for (i = j + 1; i < dimension; i++)
			if (matrix[i][j] > matrix[temp][j])
				temp = i;
		
		if (fabs(matrix[temp][j]) < minvalue) {
			printf("\n Elements are too small to deal with");
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


template<size_t rows, size_t columns>
void openmp_offload(float (&matrix)[rows][columns], float (&iden)[rows][columns], int dimension) {
	for (int i = 0; i < dimension; i++) {
		//normalize
		if (matrix[i][i] > 0) {
#pragma omp target teams distribute parallel for private(i)
			for (int y = i; y < dimension; y++) {
				iden[i][y] /= matrix[i][i];
				matrix[i][y] /= matrix[i][i];
			}
		} else {
#pragma omp target teams distribute parallel for private(i)
			for (int y = i; y < dimension; y++) {
				iden[i][y] /= -matrix[i][i];
				matrix[i][y] /= -matrix[i][i];
			}
		}
		
		/*/gauß
#pragma omp target teams distribute parallel for collapse(2) private(i)
		for (int x = 0; x < dimension; x++) {
			for (int y = 0; y < dimension; y++) {
				if (x != i) {
					iden[x][y] -= iden[i][y] * matrix[x][i];
					if (y != i) {
						matrix[x][y] -= matrix[i][y] * matrix[x][i];
					}
				}
			}
		}*/
		int temp;
#pragma omp target teams distribute parallel for shared(i) private(temp)
		for (int x = 0; x < dimension; x++) {
			temp = matrix[x][i];
			if (x != i && temp != 0) {
				for (int y = i; y < dimension; y++) {
					matrix[x][y] -= matrix[i][y] * temp;
					iden[x][y] -= matrix[i][y] * temp;
				}
			}
		}
		
		/*set 0
#pragma omp target teams distribute parallel for collapse(2) private(i)
		for (int x = 0; x < dimension; x++)
			for (int y = 0; y < dimension; y++)
				if (x != i)
					if (y == i)
						matrix[x][y] = 0;*/
	}
#pragma omp target exit data map(from: iden[dimension][dimension])
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
