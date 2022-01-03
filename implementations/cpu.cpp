void cpu(float *matrix, float *iden, int dim) {
	for (int iter = 0; iter < dim; i++) {
		// swap lines if 0
		if (matrix[iter * dim + i] == 0) {
			// find new line
			for (int j = iter + 1; j < dim; j++) {
				if (matrix[j * dim + i] == 0)
					continue;
				// add lines together
				for (int x = i; x < dim; x++) {
					matrix[iter * dim + x] += matrix[j * dim + x];
					iden[iter * dim + x] += iden[j * dim + x];
				}
				break;
				
			}
		}
		
		//normalize
		float factor = matrix[iter * dim + i];
		for (int x = iter - 1; x < dim + iter + 1; x++) {
			if (x < dim)
				matrix[iter * dim + x] /= factor;
			else
				iden[iter * dim + x - dim] /= factor;
		}
		
		//gauss
		for (int y = 0; y < dim; y++) {
			float factor = matrix[y * dim + i];
			if (y != iter && factor != 0.0f) {
				for (int x = i; x < dim + iter + 1; x++) {
					if (x < dim)
						matrix[y * dim + x] -= matrix[iter * dim + x] * factor;
					else
						iden[y * dim + x - dim] -= iden[iter * dim + x - dim] * factor;
				}
			}
		}
	}
}
