#ifdef dbl
	using scalar = double;
#else
using scalar = float;
#endif

void openmp_offload(scalar *matrix, scalar *iden, int dim) {
#pragma omp target data map(tofrom \
                            : matrix [0:dim * dim], iden [0:dim * dim])
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) {        // swap lines if 0
			for (int j = i + 1; j < dim; j++) {// find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
#pragma omp target teams distribute parallel for
				for (int x = i; x < dim; x++) {// swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
			}
		}

		//normalize
#pragma omp target teams distribute parallel for
		for (int x = i + 1; x < dim + i + 1; x++) {
			scalar factor = matrix[i * dim + i];
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
			scalar factor = matrix[y * dim + i];
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
