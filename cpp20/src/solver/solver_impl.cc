#include "solver_impl.h"

using namespace Solver;

bool SolverImpl::add_value_to_gradient_vector(const uint row_index,
                                              const double value) {
  assert(row_index < _gradient_vector.rows());
  _gradient_vector(row_index) += value;
  return true;
}

bool SolverImpl::add_to_objective_function(
    const uint start_row, const uint start_col, const uint add_row_size,
    const uint add_col_size, const Eigen::MatrixXd &add_hessian_submatrix,
    const Eigen::VectorXd &add_gradient_subvector) {
  assert(add_row_size == add_hessian_submatrix.rows());
  assert(add_col_size == add_hessian_submatrix.cols());
  _hessian_matrix.block(start_row, start_col, add_row_size, add_col_size) +=
      add_hessian_submatrix;
  // Eigen::Block<double>(_hessian_matrix, start_row, start_col, add_row_size,
  // add_col_size) += add_hessian_submatrix;
  if (add_gradient_subvector.rows() != 0) {
    // Eigen::Block<double>(_gradient_vector, start_row, start_col,
    // add_row_size, add_col_size) += add_hessian_submatrix;
    _gradient_vector.segment(start_row, add_row_size) += add_gradient_subvector;
  }
  return true;
}

bool SolverImpl::set_objective_function(
    const Eigen::MatrixXd &hessian_matrix,
    const Eigen::VectorXd &gradient_vector = Eigen::Vector<double, 0>::Zero()) {
  assert(hessian_matrix.rows() == hessian_matrix.cols() &&
         hessian_matrix.rows() == _hessian_matrix.rows());
}
