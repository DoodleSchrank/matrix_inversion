#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

void openmp_offload(scalar *matrix, scalar *iden, int dim) {
#pragma omp target data map(tofrom \
                            : matrix [0:dim * dim], iden [0:dim * dim])
	for (int iter = 0; iter < dim; iter++) {
		if (matrix[iter * dim + iter] == 0) {     // swap lines if 0
			for (int j = iter + 1; j < dim; j++) {// find new line
				if (matrix[j * dim + iter] == 0) {
					continue;
				}
#pragma omp target teams distribute parallel for simd
				for (int column = iter; column < dim; column++) {// swap lines
					matrix[iter * dim + column] += matrix[j * dim + column];
					iden[iter * dim + column] = iden[j * dim + column];
				}
				break;
			}
		}

		//normalize
#pragma omp target teams distribute parallel for
		for (int column = iter + 1; column < dim + iter + 1; column++) {
			scalar factor = matrix[iter * dim + iter];
			if (column < dim) {
				matrix[iter * dim + column] /= factor;
			} else {
				iden[iter * dim + column - dim] /= factor;
			}
		}
#pragma omp target
		matrix[iter * dim + iter] = 1.;


		//gauss
#pragma omp target teams distribute parallel for
		for (int row = 0; row < dim; row++) {
			scalar factor = matrix[row * dim + iter];
			if (row != iter && factor != 0.0f) {
#pragma omp simd
				for (int column = iter; column < dim + iter + 1; column++) {
					if (column < dim)
						matrix[row * dim + column] -= matrix[iter * dim + column] * factor;
					else
						iden[row * dim + column - dim] -= iden[iter * dim + column - dim] * factor;
				}
			}
		}
	}
}
