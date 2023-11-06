import unittest
import numpy as np
from src.geometry.polynomial2d import Polynomial2d
from src.geometry.spline1d import Spline1d
from src.geometry.spline2d import Spline2d


class TestSpline2d(unittest.TestCase):
    def test_spline_basic(self):
        # x(t) = 4t^3 + 3t^2+ 2t + 1 for [0-1]
        x_coeffs_1 = [1, 2, 3, 4]
        # x(t) = 4(t-1)^3 + 3(t-1)^2+ 2(t-1) + 10 for [1-2]
        x_coeffs_2 = [10, 2, 3, 4]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # y(t) = 5t^3 + 2 for [0-1]
        y_coeffs_1 = [2, 0, 0, 5]
        # y(t) = 5(t-1)^3 + 7 for [1-2]
        y_coeffs_2 = [7, 0, 0, 5]
        y_coeffs = [y_coeffs_1, y_coeffs_2]

        knots = [0, 1, 2]

        sp = Spline2d.from_xy_knots_and_coefficients(knots, x_coeffs, y_coeffs)

        print(sp)

        self.assertEqual(sp.knots, [0, 1, 2])
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp.order, 3)
        self.assertEqual(sp.total_param_number, 16)

        poly1 = Polynomial2d.polynomial_from_coefficients(
            x_coeffs_1, y_coeffs_1
        )
        poly2 = Polynomial2d.polynomial_from_coefficients(
            x_coeffs_2, y_coeffs_2
        )
        self.assertEqual(sp[0], poly1)
        self.assertEqual(sp[1], poly2)

        self.assertEqual(sp.line_segment(0), poly1)
        self.assertEqual(sp.line_segment(1), poly2)

        self.assertEqual(sp.knot_segment(0), [0, 1])
        self.assertEqual(sp.knot_segment(1), [1, 2])

    def test_spline_empty_error(self):
        knots = [0]
        with self.assertRaises(AssertionError):
            sp = Spline2d.from_xy_knots_and_coefficients(knots, [], [])

        # x(t) = 4t^3 + 3t^2+ 2t + 1 for [0-1]
        x_coeffs_1 = [1, 2, 3, 4]
        # x(t) = 4(t-1)^3 + 3(t-1)^2+ 2(t-1) + 10 for [1-2]
        x_coeffs_2 = [10, 2, 3, 4]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # y(t) = 5t^3 + 2 for [0-1]
        y_coeffs_1 = [2, 0, 0, 5]
        # y(t) = 5(t-1)^3 + 7 for [1-2]
        y_coeffs_2 = [7, 0, 0, 5]
        y_coeffs = [y_coeffs_1, y_coeffs_2]
        with self.assertRaises(AssertionError):
            sp = Spline2d.from_xy_knots_and_coefficients(knots, [], y_coeffs)

    def test_spline_evaluations(self):
        # x(t) = 4t^3 + 3t^2+ 2t + 1 for [0-1]
        x_coeffs_1 = [1, 2, 3, 4]
        # x(t) = 4(t-1)^3 + 3(t-1)^2+ 2(t-1) + 10 for [1-2]
        x_coeffs_2 = [10, 2, 3, 4]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # y(t) = 5t^3 + 2 for [0-1]
        y_coeffs_1 = [2, 0, 0, 5]
        # y(t) = 5(t-1)^3 + 7 for [1-2]
        y_coeffs_2 = [7, 0, 0, 5]
        y_coeffs = [y_coeffs_1, y_coeffs_2]

        knots = [0, 1, 2]
        sp = Spline2d.from_xy_knots_and_coefficients(knots, x_coeffs, y_coeffs)

        self.assertEqual(sp.evaluate(0.0), [1, 2])
        self.assertEqual(sp.evaluate(0.5), [3.25, 2.625])
        self.assertEqual(sp.evaluate(1.0), [10, 7])
        self.assertEqual(sp.evaluate(1.5), [12.25, 7.625])
        self.assertEqual(sp.evaluate(2.0), [19, 12])

        self.assertEqual(sp.derivative(0.0), [2, 0])
        self.assertEqual(sp.derivative(0.5), [8, 3.75])
        self.assertEqual(sp.derivative(1.0), [2, 0])  # Not continuous
        self.assertEqual(sp.derivative(1.5), [8, 3.75])
        self.assertEqual(sp.derivative(2.0), [20, 15])

        self.assertEqual(sp.second_derivative(0.0), [6, 0])
        self.assertEqual(sp.second_derivative(0.5), [18, 15])
        self.assertEqual(sp.second_derivative(1.0), [6, 0])  # Not continuous
        self.assertEqual(sp.second_derivative(1.5), [18, 15])
        self.assertEqual(sp.second_derivative(2.0), [30, 30])

        self.assertEqual(sp.third_derivative(0.0), [24, 30])
        self.assertEqual(sp.third_derivative(0.5), [24, 30])
        self.assertEqual(sp.third_derivative(1.0), [24, 30])
        self.assertEqual(sp.third_derivative(1.5), [24, 30])
        self.assertEqual(sp.third_derivative(2.0), [24, 30])

    def test_spline_derivative(self):
        # x(t) = 4t^3 + 3t^2+ 2t + 1 for [0-1]
        x_coeffs_1 = [1, 2, 3, 4]
        # x(t) = 4(t-1)^3 + 3(t-1)^2+ 2(t-1) + 10 for [1-2]
        x_coeffs_2 = [10, 2, 3, 4]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # y(t) = 5t^3 + 2 for [0-1]
        y_coeffs_1 = [2, 0, 0, 5]
        # y(t) = 5(t-1)^3 + 7 for [1-2]
        y_coeffs_2 = [7, 0, 0, 5]
        y_coeffs = [y_coeffs_1, y_coeffs_2]

        knots = [0, 1, 2]
        sp = Spline2d.from_xy_knots_and_coefficients(knots, x_coeffs, y_coeffs)

        # dx(t) = 12t^2+6t+2 for [0-1]
        x_coeffs_1 = [2, 6, 12]
        # dx(t) = 12(t-1)^2+6(t-1)+2 for [1-2]
        x_coeffs_2 = [2, 6, 12]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # dy(t) = 15t^2 for [0-1]
        y_coeffs_1 = [0, 0, 15]
        # dy(t) = 15(t-1)^2 for [1-2]
        y_coeffs_2 = [0, 0, 15]
        y_coeffs = [y_coeffs_1, y_coeffs_2]
        sp_de = Spline2d.from_xy_knots_and_coefficients(
            knots, x_coeffs, y_coeffs
        )
        self.assertEqual(sp.derivative_spline(1), sp_de)

    def test_spline_integral(self):
        # x(t) = 4t^3 + 3t^2+ 2t + 1 for [0-1]
        x_coeffs_1 = [1, 2, 3, 4]
        # x(t) = 4(t-1)^3 + 3(t-1)^2+ 2(t-1) + 10 for [1-2]
        x_coeffs_2 = [10, 2, 3, 4]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # y(t) = 5t^3 + 2 for [0-1]
        y_coeffs_1 = [2, 0, 0, 5]
        # y(t) = 5(t-1)^3 + 7 for [1-2]
        y_coeffs_2 = [7, 0, 0, 5]
        y_coeffs = [y_coeffs_1, y_coeffs_2]

        knots = [0, 1, 2]
        sp = Spline2d.from_xy_knots_and_coefficients(knots, x_coeffs, y_coeffs)

        # x(t) = t^4+t^3+t^2+t for [0-1]
        x_coeffs_1 = [0, 1, 1, 1, 1]
        # x(t) = (t-1)^4+(t-1)^3+(t-1)^2+10t for [1-2]
        x_coeffs_2 = [0, 10, 1, 1, 1]
        x_coeffs = [x_coeffs_1, x_coeffs_2]

        # dy(t) = 1.25t^4+2t for [0-1]
        y_coeffs_1 = [0, 2, 0, 0, 1.25]
        # dy(t) = 1.25(t-1)^4+2(t-1) for [1-2]
        y_coeffs_2 = [0, 7, 0, 0, 1.25]
        y_coeffs = [y_coeffs_1, y_coeffs_2]
        sp_int = Spline2d.from_xy_knots_and_coefficients(
            knots, x_coeffs, y_coeffs
        )
        self.assertEqual(sp.integral_spline(1), sp_int)
