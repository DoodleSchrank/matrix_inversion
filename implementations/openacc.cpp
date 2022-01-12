#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

void openacc_offload(scalar *matrix, scalar *iden, int dim) {
#pragma acc data copy(matrix [0:dim * dim], iden [0:dim * dim])
	for (int i = 0; i < dim; i++) {
		// swap lines if 0
		if (matrix[i * dim + i] == 0) {
			// find new line
			for (int j = i + 1; j < dim; j++) {
				if (matrix[j * dim + i] != 0) {
#pragma acc parallel loop worker vector
					for (int x = i; x < dim; x++) {
						matrix[i * dim + x] += matrix[j * dim + x];
						iden[i * dim + x] += iden[j * dim + x];
					}
					break;
				}
			}
		}

		//normalize

#pragma acc parallel loop gang worker vector
		for (int x = i + 1; x < dim + i + 1; x++) {
			if (x < dim)
				matrix[i * dim + x] /= matrix[i * dim + i];
			else {
				iden[i * dim + x - dim] /= matrix[i * dim + i];
			}
		}
#pragma acc serial
		matrix[i * dim + i] = 1.;

		//gauss
#pragma acc parallel loop gang worker
		for (int y = 0; y < dim; y++) {
			scalar factor = matrix[y * dim + i];
			if (y != i && factor != 0.0f) {
#pragma acc loop vector
				for (int x = i; x < dim + i + 1; x++) {
					if (x < dim) {
						matrix[y * dim + x] -= matrix[i * dim + x] * factor;
					} else {
						iden[y * dim + x - dim] -= iden[i * dim + x - dim] * factor;
					}
				}
			}
		}
	}
}
