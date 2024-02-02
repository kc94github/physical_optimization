#pragma once

#include <vector>
#include <math.h>
#include "spline_nd_solver.h"
#include "abstract.h"

namespace Solver {

class DistanceTimeSolver : public SplineNdSolver {

public:
    DistanceTimeSolver(const std::vector<double>& knots, const uint& spline_order):
        SplineNdSolver(knots, spline_order, 1) {}

    std::string toString() const {
        std::string s = "DistanceTimeSolver with knots: [";
        for (const auto &ele : _knots) {
            s += std::to_string(ele);
            s += ",";
        }
        s += "] \n";

        return s + _impl.toString();
    }

    bool add_distance_constraint(const double& t, double distance);

    bool add_speed_constraint(const double& t, double speed);

    bool add_acceleration_constraint(const double& t, double acceleration);

    bool add_jerk_constraint(const double& t, double jerk);

    bool add_distance_increasing_monotone(const std::vector<double>& t_ref = {});

    bool add_distance_point_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& points_ref);

    bool add_speed_point_penalty_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& points_ref = {});

    bool add_acceleration_point_penalty_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& points_ref = {});

    bool add_jerk_point_penalty_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& points_ref = {});

    bool add_distance_lower_bound(const double& t, double lower_bound);

    bool add_speed_lower_bound(const double& t, double lower_bound);

    bool add_acceleration_lower_bound(const double& t, double lower_bound);

    bool add_jerk_lower_bound(const double& t, double lower_bound);

    bool add_distance_upper_bound(const double& t, double upper_bound);

    bool add_speed_upper_bound(const double& t, double upper_bound);

    bool add_acceleration_upper_bound(const double& t, double upper_bound);

    bool add_jerk_upper_bound(const double& t, double upper_bound);


};
}