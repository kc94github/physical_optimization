#pragma once

#include <vector>
#include <math.h>
#include "spline_nd_solver.h"
#include "abstract.h"

namespace Solver {

class PathSolver : public SplineNdSolver {

public:
    PathSolver(const std::vector<double>& knots, const uint& spline_order):
        SplineNdSolver(knots, spline_order, 2) {}

    std::string toString() const {
        std::string s = "PathSolver with knots: [";
        for (const auto &ele : _knots) {
            s += std::to_string(ele);
            s += ",";
        }
        s += "] \n";

        return s + _impl.toString();
    }

    bool add_angle_constraint(const double& t, double angle);

    bool add_2d_boundary_constraint(const double& t, const double& ref_x, const double& ref_y, const double& ref_heading, const double& longitudinal_bound, const double& lateral_bound);


};
}