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
  return add_to_objective_function_from_sparse(
      start_row, start_col, add_row_size, add_col_size,
      add_hessian_submatrix.sparseView(), add_gradient_subvector);
}

bool SolverImpl::add_to_objective_function_from_sparse(
    const uint start_row, const uint start_col, const uint add_row_size,
    const uint add_col_size,
    const Eigen::SparseMatrix<double> &add_hessian_submatrix,
    const Eigen::VectorXd &add_gradient_subvector) {
  assert(add_row_size == add_hessian_submatrix.rows());
  assert(add_col_size == add_hessian_submatrix.cols());
  assert(start_row + add_row_size <= _size &&
         start_col + add_col_size <= _size);

  for (int k = 0; k < add_hessian_submatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(add_hessian_submatrix,
                                                       k);
         it; ++it) {
      double value = it.value();
      int row = start_row + it.row(); // row index
      int col = start_col + it.col(); // col index (here it is equal to k)
      _hessian_matrix.coeffRef(row, col) += value;
    }

  if (add_gradient_subvector.rows() != 0) {
    _gradient_vector.segment(start_row, add_row_size) += add_gradient_subvector;
  }
  return true;
}

bool SolverImpl::set_objective_function(
    const Eigen::MatrixXd &hessian_matrix,
    const Eigen::VectorXd &gradient_subvector) {
  return set_objective_function_from_sparse(hessian_matrix.sparseView(),
                                            gradient_subvector);
}

bool SolverImpl::set_objective_function_from_sparse(
    const Eigen::SparseMatrix<double> &hessian_matrix,
    const Eigen::VectorXd &gradient_vector) {
  assert(hessian_matrix.rows() == hessian_matrix.cols() &&
         hessian_matrix.rows() == _size);

  _hessian_matrix = hessian_matrix;
  if (gradient_vector.rows() == 0) {
    _gradient_vector.resize(_size);
    _gradient_vector.setZero();
  } else {
    assert(gradient_vector.rows() == _size);
    _gradient_vector = gradient_vector;
  }

  return true;
}

bool SolverImpl::set_upper_bound(const Eigen::VectorXd &upper_bound) {
  assert(upper_bound.rows() == _size);
  Eigen::VectorXd inf_lower_bound(_size);
  inf_lower_bound.setConstant(std::numeric_limits<double>::min());

  return add_constraint_with_bounds(
      Eigen::MatrixXd::Identity(_size, _size).sparseView(), upper_bound,
      inf_lower_bound);
}

bool SolverImpl::set_lower_bound(const Eigen::VectorXd &lower_bound) {
  assert(lower_bound.rows() == _size);
  Eigen::VectorXd inf_upper_bound(_size);
  inf_upper_bound.setConstant(std::numeric_limits<double>::max());

  return add_constraint_with_bounds(
      Eigen::MatrixXd::Identity(_size, _size).sparseView(), inf_upper_bound,
      lower_bound);
}

bool SolverImpl::add_equality_constraint_with_sparse(
    const Eigen::SparseMatrix<double> &equality_matrix,
    const Eigen::VectorXd &equality_vector) {
  return add_constraint_with_bounds(equality_matrix, equality_vector,
                                    equality_vector);
}

bool SolverImpl::add_equality_constraint(
    const Eigen::MatrixXd &equality_matrix,
    const Eigen::VectorXd &equality_vector) {
  return add_constraint_with_bounds(equality_matrix.sparseView(),
                                    equality_vector, equality_vector);
}

bool SolverImpl::add_inequality_constraint_with_sparse(
    const Eigen::SparseMatrix<double> &inequality_matrix,
    const Eigen::VectorXd &inequality_vector) {
  Eigen::VectorXd lower_bound(inequality_matrix.rows());
  lower_bound.setConstant(std::numeric_limits<double>::min());
  return add_constraint_with_bounds(inequality_matrix, inequality_vector,
                                    lower_bound);
}

bool SolverImpl::add_inequality_constraint(
    const Eigen::MatrixXd &inequality_matrix,
    const Eigen::VectorXd &inequality_vector) {
  Eigen::VectorXd lower_bound(inequality_matrix.rows());
  lower_bound.setConstant(std::numeric_limits<double>::min());
  return add_constraint_with_bounds(inequality_matrix.sparseView(),
                                    inequality_vector, lower_bound);
}

bool SolverImpl::add_constraint_with_bounds(
    const Eigen::SparseMatrix<double> &constraint_matrix,
    const Eigen::VectorXd &upper_bound, const Eigen::VectorXd &lower_bound) {
  assert(constraint_matrix.rows() == upper_bound.rows());
  assert(constraint_matrix.rows() == lower_bound.rows());
  assert(constraint_matrix.cols() == _size);

  const int prev_constraint_size = _total_constraint_matrix.rows();

  _total_constraint_matrix.conservativeResize(
      prev_constraint_size + constraint_matrix.rows(), _size);

  // Append matB to matA in the row direction
  for (int k = 0; k < constraint_matrix.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(constraint_matrix, k);
         it; ++it) {
      std::cout << it.row() << ", " << it.col() << ", " << it.value()
                << std::endl;
      _total_constraint_matrix.insert(prev_constraint_size + it.row(),
                                      it.col()) = it.value();
    }
  }

  _total_constraint_upper_bound.conservativeResize(prev_constraint_size +
                                                   constraint_matrix.rows());
  _total_constraint_upper_bound.segment(prev_constraint_size,
                                        constraint_matrix.rows()) = upper_bound;

  _total_constraint_lower_bound.conservativeResize(prev_constraint_size +
                                                   constraint_matrix.rows());
  _total_constraint_lower_bound.segment(prev_constraint_size,
                                        constraint_matrix.rows()) = lower_bound;

  return true;
}

Eigen::VectorXd SolverImpl::solve() {
  OsqpEigen::Solver solver;
  // settings
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);

  // set the initial data of the QP solver
  solver.data()->setNumberOfVariables(_size);

  solver.data()->setHessianMatrix(_hessian_matrix);
  solver.data()->setGradient(_gradient_vector);

  // int total_constraint_size = _equality_constraint_matrix.rows() +
  // _inequality_constraint_matrix.rows() + _size;
  solver.data()->setNumberOfConstraints(_total_constraint_matrix.rows());
  // Eigen::Vector<double, Eigen::Dynamic> constraint_upper_bound,
  // constraint_lower_bound; _summarize_constraint_matrix(constraint_matrix,
  // constraint_upper_bound, constraint_lower_bound);
  solver.data()->setLinearConstraintsMatrix(_total_constraint_matrix);

  solver.data()->setLowerBound(_total_constraint_lower_bound);
  solver.data()->setUpperBound(_total_constraint_upper_bound);

  solver.initSolver();

  solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError;
  Eigen::Vector<double, Eigen::Dynamic> QPSolution = solver.getSolution();
  return QPSolution;
}
