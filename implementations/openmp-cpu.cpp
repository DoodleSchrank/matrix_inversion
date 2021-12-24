#include <omp.h>

#ifdef dbl
	using scalar = double;
#else
	using scalar = float;
#endif

void openmp_cpu(scalar *matrix, scalar *iden, int dim) {
	for (int i = 0; i < dim; i++) {
		if (matrix[i * dim + i] == 0) { // swap lines if 0
			for (int j = i + 1; j < dim; j++) { // find new line
				if (matrix[j * dim + i] == 0) {
					continue;
				}
#pragma omp parallel for
				for (int x = i; x < dim; x++) { // swap lines
					matrix[i * dim + x] += matrix[j * dim + x];
					iden[i * dim + x] = iden[j * dim + x];
				}
				break;
				
			}
		}
		
		//normalize
		scalar factor = matrix[i * dim + i];
		for (int x = 0; x < 2 * dim; x++) {
			if (x < dim)
				matrix[i * dim + x] /= factor;
			else
				iden[i * dim + x - dim] /= factor;
		}
		
		//gauss
		for (int y = 0; y < dim; y++) {
			scalar factor = matrix[y * dim + i];
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
