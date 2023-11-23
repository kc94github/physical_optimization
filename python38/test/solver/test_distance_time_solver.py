import unittest
import numpy as np
from typing import List, Tuple
from src.solver.distance_time_solver import DistanceTimeSolver


class TestDistanceTimeSolver(unittest.TestCase):
    def test_knots(self):
        return [0, 1, 2]

    def test_spline_order(self):
        return 3

    def test_dimension(self):
        return 1

    def test_basic(self):
        s = DistanceTimeSolver(
            self.test_knots(), self.test_spline_order()
        )
        print(s)

        self.assertEqual(len(s), 8)
        self.assertEqual(s.total_param_length, 8)

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
        self.assertAlmostEqual(solution[4], 10, precision)
        self.assertAlmostEqual(solution[5], 20, precision)
        self.assertAlmostEqual(solution[6], 15, precision)
        self.assertAlmostEqual(solution[7], 4, precision)


    def test_eval(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
        s = DistanceTimeSolver.from_knots_and_order(self.test_knots(), self.test_spline_order())
        for t, x in zip(self.test_t(), self.test_x()):
            s.add_distance_constraint(t, x)

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst)

    def test_eval_derivatives(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
        s = DistanceTimeSolver.from_knots_and_order(self.test_knots(), self.test_spline_order())
        for t, x in zip(self.test_t(), self.test_x()):
            s.add_point_constraint(t, x)

        for t, dx in zip(
            self.test_t(),
            self.test_x_first_derivative()
        ):
            s.add_speed_constraint(t, dx)

        for t, ddx in zip(
            self.test_t(),
            self.test_x_second_derivative()
        ):
            s.add_acceleration_constraint(t, ddx)

        for t, dddx in zip(
            self.test_t(),
            self.test_x_third_derivative()
        ):
            s.add_jerk_constraint(t, dddx)

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst)

    def test_lower_and_upper_bound(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]

        # Y(t) = t^3 + t + 5 for [0-1]
        # Y(t) = (t-1)^3 + 3(t-1)^2 + 4(t-1) + 7 for [1-2]
        s = DistanceTimeSolver.from_knots_and_order(self.test_knots(), self.test_spline_order())
        for t, x in zip(self.test_t(), self.test_x()):
            s.add_distance_constraint(t, x)
            s.add_distance_lower_bound(t, x - 0.1)  # Test upper bound and lower bound that satisfy the condition
            s.add_distance_upper_bound(t, x + 0.1)

        for t, dx in zip(
            self.test_t(),
            self.test_x_first_derivative()
        ):
            s.add_speed_constraint(t, dx)
            s.add_speed_lower_bound(t, dx - 0.1)
            s.add_speed_upper_bound(t, dx + 0.1)

        for t, ddx in zip(
            self.test_t(),
            self.test_x_second_derivative()
        ):
            s.add_acceleration_constraint(t, ddx)
            s.add_acceleration_lower_bound(
                t, ddx - 0.1
            )
            s.add_acceleration_upper_bound(
                t, ddx + 0.1
            )

        for t, dddx in zip(
            self.test_t(),
            self.test_x_third_derivative()
        ):
            s.add_jerk_constraint(t, dddx)
            s.add_jerk_lower_bound(
                t, dddx - 0.1
            )
            s.add_jerk_upper_bound(
                t, dddx + 0.1
            )

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst, 3)

    def test_hessian_with_derivatives(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
        s = DistanceTimeSolver.from_knots_and_order(self.test_knots(), self.test_spline_order())
        t = np.array(self.test_t())
        pts = np.array(self.test_x())

        s.add_distance_point_to_objective(100000, t, pts)
        s.add_speed_point_penalty_to_objective(
            0.0001, t
        )  # Avoid changes in curves
        s.add_acceleration_point_penalty_to_objective(0.0001, t)
        s.add_jerk_point_penalty_to_objective(0.0001, t)
        rst = s.solve()
        self.assertThreeKnotsThirdOrderEquation(rst, 1)

    def test_smooth_constraint(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
        s = DistanceTimeSolver.from_knots_and_order(self.test_knots(), self.test_spline_order())
        for t, x in zip(self.test_t(), self.test_x()):
            s.add_distance_constraint(t, x)

        for t, dx in zip(
            self.test_t(),
            self.test_x_first_derivative()
        ):
            s.add_speed_constraint(t, dx)

        for t, ddx in zip(
            self.test_t(),
            self.test_x_second_derivative()
        ):
            s.add_acceleration_constraint(t, ddx)

        for t, dddx in zip(
            self.test_t(),
            self.test_x_third_derivative()
        ):
            s.add_jerk_constraint(t, dddx)

        # Adding smooth constraint across all knots
        self.assertEqual(
            s.add_smooth_constraint(order=3), True
        )  # order = 3, including all smooth constraint of order 0, 1, 2

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst)

    def test_monotone(self):
        # X(t) = 4t^3+3t^2+2t+1 for [0-1]
        # X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
        s = DistanceTimeSolver.from_knots_and_order(self.test_knots(), self.test_spline_order())
        for t, x in zip(self.test_t(), self.test_x()):
            s.add_distance_constraint(t, x)
        self.assertEqual(s.add_distance_increasing_monotone(), True)

        rst = s.solve()
        self.assertEqual(type(rst), np.ndarray)
        self.assertThreeKnotsThirdOrderEquation(rst)

