void cpu(float *matrix, float *iden, int dim) {
	for (int iter = 0; iter < dim; i++) {
		// swap lines if 0
		if (matrix[iter * dim + iter] == 0) {
			// find new line
			for (int j = iter + 1; j < dim; j++) {
				if (matrix[j * dim + iter] == 0)
					continue;
				// add lines together
				for (int column = i; column < dim; column++) {
					matrix[iter * dim + column] += matrix[j * dim + column];
					iden[iter * dim + column] += iden[j * dim + column];
				}
				break;
				
			}
		}
		
		//normalize
		float divisor = matrix[iter * dim + iter];
		for (int column = iter; column < dim + iter + 1; column++) {
			if (column < dim)
				matrix[iter * dim + column] /= divisor;
			else
				iden[iter * dim + column - dim] /= divisor;
		}
		
		//gauss
		for (int row = 0; row < dim; row++) {
			float factor = matrix[row * dim + iter];
			if (row != iter && factor != 0.0f) {
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
