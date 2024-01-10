#include "spline_nd_solver.h"

using namespace Solver;

SplineNdSolver::SplineNdSolver(const std::vector<double> &knots,
                               const uint &spline_order, const uint &dimension)
    : Geometry::CoefficientBase(spline_order), _knots(knots),
      _spline_order(spline_order), _dimension(dimension),
      _param_size(_dimension * (_spline_order + 1) * (_knots.size() - 1)),
      _impl(Solver::SolverImpl(_param_size)) {
  assert(_knots.size() >= 2);
}

std::string SplineNdSolver::toString() const {
  std::string s = "SplineNdSolver with knots: [";
  for (const auto &ele : _knots) {
    s += std::to_string(ele);
    s += ",";
  }
  s += "] \n";

  return s + _impl.toString();
}

void SplineNdSolver::add_regularization(const double &regularization_param) {
  Eigen::SparseMatrix<double> &hes = _impl.hessian_matrix();
  for (int i = 0; i < hes.rows(); i++) {
    hes.coeffRef(i, i) += regularization_param;
  }
}

// bool SplineNdSolver::add_point_constraint(const double& t, const
// std::vector<double>& point) {
//     // return get_knot_index_and_coefficient(&CoefficientBase::t_coefficient,
//     t); return
//     apply_equality_constraint(std::bind(&SplineNdSolver::get_knot_index_and_coefficient,
//     this, std::placeholders::_1, std::placeholders::_2),
//     &CoefficientBase::t_coefficient, t, point);
// }

bool SplineNdSolver::add_point_constraint(const double &t,
                                          const std::vector<double> &point) {
  // auto knotIndexAndCoefficientFunc = std::bind(
  //     &SplineNdSolver::get_knot_index_and_coefficient<decltype(&CoefficientBase::t_coefficient)>,
  //     this,
  //     &CoefficientBase::t_coefficient,
  //     std::placeholders::_1
  // );

  // return apply_equality_constraint(knotIndexAndCoefficientFunc, t, point);

  return apply_equality_constraint(
      std::bind(&SplineNdSolver::get_knot_index_and_coefficient<decltype(
                    &CoefficientBase::t_coefficient)>,
                this, &CoefficientBase::t_coefficient, std::placeholders::_1),
      t, point);
}

bool SplineNdSolver::add_point_first_derivative_constraint(
    const double &t, const std::vector<double> &point) {
  return apply_equality_constraint(
      std::bind(&SplineNdSolver::get_knot_index_and_coefficient<decltype(
                    &CoefficientBase::t_first_derivative_coefficient)>,
                this, &CoefficientBase::t_first_derivative_coefficient,
                std::placeholders::_1),
      t, point);
}

bool SplineNdSolver::add_point_second_derivative_constraint(
    const double &t, const std::vector<double> &point) {
  return apply_equality_constraint(
      std::bind(&SplineNdSolver::get_knot_index_and_coefficient<decltype(
                    &CoefficientBase::t_second_derivative_coefficient)>,
                this, &CoefficientBase::t_second_derivative_coefficient,
                std::placeholders::_1),
      t, point);
}

bool SplineNdSolver::add_point_third_derivative_constraint(
    const double &t, const std::vector<double> &point) {
  return apply_equality_constraint(
      std::bind(&SplineNdSolver::get_knot_index_and_coefficient<decltype(
                    &CoefficientBase::t_third_derivative_coefficient)>,
                this, &CoefficientBase::t_third_derivative_coefficient,
                std::placeholders::_1),
      t, point);
}

bool SplineNdSolver::_add_smooth_constraint_order_zero() {
  assert(_knots.size() >= 3);
  Eigen::MatrixXd smooth_matrix_A =
      Eigen::MatrixXd::Zero(_dimension * (_knots.size() - 2), _param_size);
  Eigen::VectorXd smooth_matrix_B =
      Eigen::VectorXd::Zero(_dimension * (_knots.size() - 2));

  for (uint i = 0; i < _knots.size() - 2; i++) {
    // Current connection knot
    // Since we are using relative_t for a knot, we need the relative_t to use
    // the coefficient function as well
    double relative_t = _knots[i + 1] - _knots[i];
    std::vector<double> coeff = t_coefficient(relative_t);
    uint index_offset = i * _dimension * (_spline_order + 1);
    uint next_index_offset = (i + 1) * _dimension * (_spline_order + 1);
    for (uint d = 0; d < _dimension; d++) {
      for (uint j = 0; j < _spline_order + 1; j++) {
        smooth_matrix_A(i * _dimension + d,
                        index_offset + d * (_spline_order + 1) + j) = coeff[j];
      }
      smooth_matrix_A(i * _dimension + d,
                      next_index_offset + d * (_spline_order + 1)) = -1;
    }
  }

  return _impl.add_equality_constraint(smooth_matrix_A, smooth_matrix_B);
}

bool SplineNdSolver::_add_smooth_constraint_order_one() {
  assert(_knots.size() >= 3);
  Eigen::MatrixXd smooth_matrix_A =
      Eigen::MatrixXd::Zero(_dimension * (_knots.size() - 2), _param_size);
  Eigen::VectorXd smooth_matrix_B =
      Eigen::VectorXd::Zero(_dimension * (_knots.size() - 2));

  for (uint i = 0; i < _knots.size() - 2; i++) {
    // Current connection knot
    // Since we are using relative_t for a knot, we need the relative_t to use
    // the coefficient function as well
    double relative_t = _knots[i + 1] - _knots[i];
    std::vector<double> coeff = t_first_derivative_coefficient(relative_t);
    uint index_offset = i * _dimension * (_spline_order + 1);
    uint next_index_offset = (i + 1) * _dimension * (_spline_order + 1);
    for (uint d = 0; d < _dimension; d++) {
      for (uint j = 0; j < _spline_order + 1; j++) {
        smooth_matrix_A(i * _dimension + d,
                        index_offset + d * (_spline_order + 1) + j) = coeff[j];
      }
      smooth_matrix_A(i * _dimension + d,
                      next_index_offset + d * (_spline_order + 1) + 1) = -1;
    }
  }

  return _impl.add_equality_constraint(smooth_matrix_A, smooth_matrix_B);
}

bool SplineNdSolver::_add_smooth_constraint_order_two() {
  assert(_knots.size() >= 3);
  Eigen::MatrixXd smooth_matrix_A =
      Eigen::MatrixXd::Zero(_dimension * (_knots.size() - 2), _param_size);
  Eigen::VectorXd smooth_matrix_B =
      Eigen::VectorXd::Zero(_dimension * (_knots.size() - 2));

  for (uint i = 0; i < _knots.size() - 2; i++) {
    // Current connection knot
    // Since we are using relative_t for a knot, we need the relative_t to use
    // the coefficient function as well
    double relative_t = _knots[i + 1] - _knots[i];
    std::vector<double> coeff = t_second_derivative_coefficient(relative_t);
    uint index_offset = i * _dimension * (_spline_order + 1);
    uint next_index_offset = (i + 1) * _dimension * (_spline_order + 1);
    for (uint d = 0; d < _dimension; d++) {
      for (uint j = 0; j < _spline_order + 1; j++) {
        smooth_matrix_A(i * _dimension + d,
                        index_offset + d * (_spline_order + 1) + j) = coeff[j];
      }
      smooth_matrix_A(i * _dimension + d,
                      next_index_offset + d * (_spline_order + 1) + 2) = -2;
    }
  }

  return _impl.add_equality_constraint(smooth_matrix_A, smooth_matrix_B);
}

bool SplineNdSolver::_add_smooth_constraint_order_three() {
  assert(_knots.size() >= 3);
  Eigen::MatrixXd smooth_matrix_A =
      Eigen::MatrixXd::Zero(_dimension * (_knots.size() - 2), _param_size);
  Eigen::VectorXd smooth_matrix_B =
      Eigen::VectorXd::Zero(_dimension * (_knots.size() - 2));

  for (uint i = 0; i < _knots.size() - 2; i++) {
    // Current connection knot
    // Since we are using relative_t for a knot, we need the relative_t to use
    // the coefficient function as well
    double relative_t = _knots[i + 1] - _knots[i];
    std::vector<double> coeff = t_third_derivative_coefficient(relative_t);
    uint index_offset = i * _dimension * (_spline_order + 1);
    uint next_index_offset = (i + 1) * _dimension * (_spline_order + 1);
    for (uint d = 0; d < _dimension; d++) {
      for (uint j = 0; j < _spline_order + 1; j++) {
        smooth_matrix_A(i * _dimension + d,
                        index_offset + d * (_spline_order + 1) + j) = coeff[j];
      }
      smooth_matrix_A(i * _dimension + d,
                      next_index_offset + d * (_spline_order + 1) + 3) = -6;
    }
  }

  return _impl.add_equality_constraint(smooth_matrix_A, smooth_matrix_B);
}
