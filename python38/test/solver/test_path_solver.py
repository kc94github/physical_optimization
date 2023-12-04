import unittest
import numpy as np
import math
from typing import List, Tuple
from src.solver.path_solver import PathSolver


class TestPathSolver(unittest.TestCase):
    def test_knots(self):
        return [0, 1, 2]

    def test_spline_order(self):
        return 3

    def test_dimension(self):
        return 2

    def test_basic(self):
        s = PathSolver(self.test_knots(), self.test_spline_order())
        print(s)

        self.assertEqual(len(s), 16)
        self.assertEqual(s.total_param_length, 16)

    def test_t(self):
        return [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    def test_x(self):
        return [1.0, 1.75, 3.25, 5.875, 10.0, 16.0, 24.25, 35.125, 49.0]

    def test_x_first_derivative(self):
        return [2.0, 4.25, 8.0, 13.25, 20.0, 28.25, 38.0, 49.25, 62.0]

    def test_x_second_derivative(self):
        return [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0]

    def test_x_third_derivative(self):
        return [24, 24, 24, 24, 24, 24, 24, 24, 24]

    def test_y(self):
        return [
            5.0,
            5.265625,
            5.625,
            6.171875,
            7.0,
            8.203125,
            9.875,
            12.109375,
            15.0,
        ]

    def test_pts(self):
        x = self.test_x()
        y = self.test_y()
        return [[xt, yt] for xt, yt in zip(x, y)]

    def test_y_first_derivative(self):
        return [1.0, 1.1875, 1.75, 2.6875, 4.0, 5.6875, 7.75, 10.1875, 13.0]

    def test_y_second_derivative(self):
        return [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0]

    def test_y_third_derivative(self):
        return [6, 6, 6, 6, 6, 6, 6, 6, 6]

    def assertThreeKnotsThirdOrderEquation(
        self, solution: np.ndarray, precision=5
    ):
        self.assertAlmostEqual(solution[0], 1, precision)
        self.assertAlmostEqual(solution[1], 2, precision)
        self.assertAlmostEqual(solution[2], 3, precision)
        self.assertAlmostEqual(solution[3], 4, precision)
        self.assertAlmostEqual(solution[4], 5, precision)
        self.assertAlmostEqual(solution[5], 1, precision)
        self.assertAlmostEqual(solution[6], 0, precision)
        self.assertAlmostEqual(solution[7], 1, precision)
        self.assertAlmostEqual(solution[8], 10, precision)
        self.assertAlmostEqual(solution[9], 20, precision)
        self.assertAlmostEqual(solution[10], 15, precision)
        self.assertAlmostEqual(solution[11], 4, precision)
        self.assertAlmostEqual(solution[12], 7, precision)
        self.assertAlmostEqual(solution[13], 4, precision)
        self.assertAlmostEqual(solution[14], 3, precision)
        self.assertAlmostEqual(solution[15], 1, precision)

    def test_2d_boundary_constraint(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]

        # Y(t) = t^3 + t + 5 for [0-1]
        # Y(t) = (t-1)^3 + 3(t-1)^2 + 4(t-1) + 7 for [1-2]
        s = PathSolver(self.test_knots(), self.test_spline_order())

        longitudinal_bound = lateral_bound = 0.01
        for t, x, y, dx, dy in zip(
            self.test_t(),
            self.test_x(),
            self.test_y(),
            self.test_x_first_derivative(),
            self.test_y_first_derivative(),
        ):
            s.add_point_constraint(t, x, y)
            heading = math.atan2(dy, dx)
            s.add_2d_boundary_constraint(
                t, x, y, heading, longitudinal_bound, lateral_bound
            )

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst)

    def test_heading_constraint(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]

        # Y(t) = t^3 + t + 5 for [0-1]
        # Y(t) = (t-1)^3 + 3(t-1)^2 + 4(t-1) + 7 for [1-2]
        s = PathSolver(self.test_knots(), self.test_spline_order())

        longitudinal_bound = lateral_bound = 0.01
        for t, x, y, dx, dy in zip(
            self.test_t(),
            self.test_x(),
            self.test_y(),
            self.test_x_first_derivative(),
            self.test_y_first_derivative(),
        ):
            s.add_point_constraint(t, x, y)
            heading = math.atan2(dy, dx)
            s.add_2d_boundary_constraint(
                t, x, y, heading, longitudinal_bound, lateral_bound
            )
            s.add_angle_constraint(t, heading)

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst)
