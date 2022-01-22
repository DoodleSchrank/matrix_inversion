#include <Eigen/Core>
#include <Eigen/Dense>

#ifdef dbl
using scalar = double;
#else
using scalar = float;
#endif

typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

void eigen(scalar *matrix, int dim) {
	Eigen::Map<MatrixXs>eigenMatrix(matrix, dim, dim) ;
	Eigen::Map<MatrixXs>(matrix, dim, dim) = eigenMatrix.inverse();
}