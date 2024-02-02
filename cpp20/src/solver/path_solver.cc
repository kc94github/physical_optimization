#include "path_solver.h"

namespace Solver {

bool PathSolver::add_angle_constraint(const double &t, double angle) {
  uint index = _search_prev_knot_index(t);
  double relative_time = t - _knots[index];
  std::vector<double> coeff = t_first_derivative_coefficient(relative_time);

  double sin = std::sin(angle);
  double cos = std::cos(angle);

  Eigen::MatrixXd matrix_A = Eigen::MatrixXd::Zero(1, _param_size);
  Eigen::VectorXd matrix_B = Eigen::VectorXd::Zero(1);

  uint offset_index = _dimension * (_spline_order + 1) * index;
  for (uint i = 0; i < _spline_order + 1; i++) {

    matrix_A(0, offset_index + i) = -sin * coeff[i];
    matrix_A(0, offset_index + i + _spline_order + 1) = cos * coeff[i];
  }

  if (!_impl.add_equality_constraint(matrix_A, matrix_B))
    return false;

  while (angle < 0) {
    angle += 2 * M_PI;
  }

  while (angle >= 2 * M_PI) {
    angle -= 2 * M_PI;
  }

  Eigen::MatrixXd inequality_matrix_A = Eigen::MatrixXd::Zero(2, _param_size);
  Eigen::VectorXd inequality_matrix_B = Eigen::VectorXd::Zero(2);

  int x_sign = 1, y_sign = 1;
  if (angle > M_PI / 2.0 && angle < M_PI * 1.5) {
    x_sign = -1;
  }
  if (angle > M_PI) {
    y_sign = -1;
  }

  for (uint i = 0; i < _spline_order + 1; i++) {
    inequality_matrix_A(0, offset_index + i) = coeff[i] * -x_sign;
    inequality_matrix_A(1, offset_index + i + _spline_order + 1) =
        coeff[i] * -y_sign;
  }

  return _impl.add_inequality_constraint(inequality_matrix_A,
                                         inequality_matrix_B);
}

bool PathSolver::add_2d_boundary_constraint(const double &t,
                                            const double &ref_x,
                                            const double &ref_y,
                                            const double &ref_heading,
                                            const double &longitudinal_bound,
                                            const double &lateral_bound) {
  uint index = _search_prev_knot_index(t);
  double relative_time = t - _knots[index];
  std::vector<double> coeff = t_coefficient(relative_time);

  const double sin = std::sin(ref_heading);
  const double cos = std::cos(ref_heading);

  Eigen::MatrixXd inequality_matrix_A = Eigen::MatrixXd::Zero(4, _param_size);
  Eigen::VectorXd inequality_matrix_B = Eigen::VectorXd::Zero(4);

  inequality_matrix_B(0) = longitudinal_bound + cos * ref_x + sin * ref_y;
  inequality_matrix_B(1) = longitudinal_bound - cos * ref_x - sin * ref_y;
  inequality_matrix_B(2) = lateral_bound - sin * ref_x + cos * ref_y;
  inequality_matrix_B(3) = lateral_bound + sin * ref_x - cos * ref_y;

  uint offset_index = _dimension * (_spline_order + 1) * index;

  for (uint i = 0; i < _spline_order + 1; i++) {

    inequality_matrix_A(0, offset_index + i) = cos * coeff[i];
    inequality_matrix_A(0, offset_index + _spline_order + 1 + i) =
        sin * coeff[i];

    inequality_matrix_A(1, offset_index + i) = -cos * coeff[i];
    inequality_matrix_A(1, offset_index + _spline_order + 1 + i) =
        -sin * coeff[i];

    inequality_matrix_A(2, offset_index + i) = -sin * coeff[i];
    inequality_matrix_A(2, offset_index + _spline_order + 1 + i) =
        cos * coeff[i];

    inequality_matrix_A(3, offset_index + i) = sin * coeff[i];
    inequality_matrix_A(3, offset_index + _spline_order + 1 + i) =
        -cos * coeff[i];
  }

  return _impl.add_inequality_constraint(inequality_matrix_A,
                                         inequality_matrix_B);
}

} // namespace Solver