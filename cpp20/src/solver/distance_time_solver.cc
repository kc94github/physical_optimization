#include "distance_time_solver.h"

namespace Solver {

bool DistanceTimeSolver::add_distance_constraint(const double &t,
                                                 double distance) {
  return add_point_constraint(t, {distance});
}

bool DistanceTimeSolver::add_speed_constraint(const double &t, double speed) {
  return add_point_first_derivative_constraint(t, {speed});
}

bool DistanceTimeSolver::add_acceleration_constraint(const double &t,
                                                     double acceleration) {
  return add_point_second_derivative_constraint(t, {acceleration});
}

bool DistanceTimeSolver::add_jerk_constraint(const double &t, double jerk) {
  return add_point_third_derivative_constraint(t, {jerk});
}

bool DistanceTimeSolver::add_distance_point_to_objective(
    const double &weight, const std::vector<double> &t_ref,
    const std::vector<std::vector<double>> &points_ref) {
  return add_reference_points_to_objective(weight, t_ref, points_ref);
}

bool DistanceTimeSolver::add_speed_point_penalty_to_objective(
    const double &weight, const std::vector<double> &t_ref,
    const std::vector<std::vector<double>> &points_ref) {
  return add_first_derivative_points_to_objective(weight, t_ref, points_ref);
}

bool DistanceTimeSolver::add_acceleration_point_penalty_to_objective(
    const double &weight, const std::vector<double> &t_ref,
    const std::vector<std::vector<double>> &points_ref) {
  return add_second_derivative_points_to_objective(weight, t_ref, points_ref);
}

bool DistanceTimeSolver::add_jerk_point_penalty_to_objective(
    const double &weight, const std::vector<double> &t_ref,
    const std::vector<std::vector<double>> &points_ref) {
  return add_third_derivative_points_to_objective(weight, t_ref, points_ref);
}

bool DistanceTimeSolver::add_distance_increasing_monotone(
    const std::vector<double> &t_ref) {
  if (t_ref.size() == 0) {
    bool rst = true;
    for (const auto &val : _knots) {
      rst = rst && add_point_first_derivative_lower_bound(val, {0.0});
    }
    return rst;
  } else {
    bool rst = true;
    for (const auto &val : t_ref) {
      rst = rst && add_point_first_derivative_lower_bound(val, {0.0});
    }
    return rst;
  }
  return true;
}

bool DistanceTimeSolver::add_distance_lower_bound(const double &t,
                                                  double lower_bound) {
  return add_point_lower_bound(t, {lower_bound});
}

bool DistanceTimeSolver::add_speed_lower_bound(const double &t,
                                               double lower_bound) {
  return add_point_first_derivative_lower_bound(t, {lower_bound});
}

bool DistanceTimeSolver::add_acceleration_lower_bound(const double &t,
                                                      double lower_bound) {
  return add_point_second_derivative_lower_bound(t, {lower_bound});
}

bool DistanceTimeSolver::add_jerk_lower_bound(const double &t,
                                              double lower_bound) {
  return add_point_third_derivative_lower_bound(t, {lower_bound});
}

bool DistanceTimeSolver::add_distance_upper_bound(const double &t,
                                                  double upper_bound) {
  return add_point_upper_bound(t, {upper_bound});
}

bool DistanceTimeSolver::add_speed_upper_bound(const double &t,
                                               double upper_bound) {
  return add_point_first_derivative_upper_bound(t, {upper_bound});
}

bool DistanceTimeSolver::add_acceleration_upper_bound(const double &t,
                                                      double upper_bound) {
  return add_point_second_derivative_upper_bound(t, {upper_bound});
}

bool DistanceTimeSolver::add_jerk_upper_bound(const double &t,
                                              double upper_bound) {
  return add_point_third_derivative_upper_bound(t, {upper_bound});
}

} // namespace Solver