import unittest
import numpy as np
from src.geometry.polynomial1d import Polynomial1d
from src.geometry.spline1d import Spline1d


class TestSpline1d(unittest.TestCase):
    def test_spline_basic(self):
        # y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
        coeffs_1 = [1, 2, 3, 4]
        # y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
        coeffs_2 = [10, 2, 3, 4]

        knots = [0, 1, 2]

        poly1 = Polynomial1d.polynomial_from_coefficients(coeffs_1)
        poly2 = Polynomial1d.polynomial_from_coefficients(coeffs_2)

        sp = Spline1d.spline_from_knots_and_polynomials(knots, [poly1, poly2])

        print(sp)

        self.assertEqual(sp.knots, [0, 1, 2])
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp.order, 3)
        self.assertEqual(sp.total_param_number, 8)

        self.assertEqual(sp[0], poly1)
        self.assertEqual(sp[1], poly2)
        self.assertEqual(sp.line_segment(0), poly1)
        self.assertEqual(sp.line_segment(1), poly2)

        self.assertEqual(sp.knot_segment(0), [0, 1])
        self.assertEqual(sp.knot_segment(1), [1, 2])

    def test_spline_empty_error(self):
        knots = [0]
        with self.assertRaises(AssertionError):
            sp = Spline1d.spline_from_knots_and_polynomials(knots, [])

    def test_spline_search_knot(self):
        # y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
        coeffs_1 = [1, 2, 3, 4]
        # y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
        coeffs_2 = [10, 2, 3, 4]

        knots = [0, 1, 2]

        poly1 = Polynomial1d.polynomial_from_coefficients(coeffs_1)
        poly2 = Polynomial1d.polynomial_from_coefficients(coeffs_2)

        sp = Spline1d.spline_from_knots_and_polynomials(knots, [poly1, poly2])

        with self.assertRaises(Exception):
            sp.search_prev_knot_index(-0.5)
        self.assertEqual(sp.search_prev_knot_index(0.0), 0)
        self.assertEqual(sp.search_prev_knot_index(0.5), 0)
        self.assertEqual(sp.search_prev_knot_index(1.0), 1)
        self.assertEqual(sp.search_prev_knot_index(1.5), 1)
        self.assertEqual(sp.search_prev_knot_index(2.0), 1)
        self.assertEqual(sp.search_prev_knot_index(2.5), 1)

    def test_spline_evaluations(self):
        # y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
        coeffs_1 = [1, 2, 3, 4]
        # y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
        coeffs_2 = [10, 2, 3, 4]

        knots = [0, 1, 2]

        poly1 = Polynomial1d.polynomial_from_coefficients(coeffs_1)
        poly2 = Polynomial1d.polynomial_from_coefficients(coeffs_2)

        sp = Spline1d.spline_from_knots_and_polynomials(knots, [poly1, poly2])

        self.assertEqual(sp.evaluate(0.0), 1)
        self.assertEqual(sp.evaluate(0.5), 3.25)
        self.assertEqual(sp.evaluate(1.0), 10)
        self.assertEqual(sp.evaluate(1.5), 12.25)
        self.assertEqual(sp.evaluate(2.0), 19)

        self.assertEqual(sp.derivative(0.0), 2)
        self.assertEqual(sp.derivative(0.5), 8)
        self.assertEqual(sp.derivative(1.0), 2)  # Not continuous
        self.assertEqual(sp.derivative(1.5), 8)
        self.assertEqual(sp.derivative(2.0), 20)

        self.assertEqual(sp.second_derivative(0.0), 6)
        self.assertEqual(sp.second_derivative(0.5), 18)
        self.assertEqual(sp.second_derivative(1.0), 6)  # Not continuous
        self.assertEqual(sp.second_derivative(1.5), 18)
        self.assertEqual(sp.second_derivative(2.0), 30)

        self.assertEqual(sp.third_derivative(0.0), 24)
        self.assertEqual(sp.third_derivative(0.5), 24)
        self.assertEqual(sp.third_derivative(1.0), 24)
        self.assertEqual(sp.third_derivative(1.5), 24)
        self.assertEqual(sp.third_derivative(2.0), 24)

    def test_spline_derivative(self):
        # y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
        coeffs_1 = [1, 2, 3, 4]
        coeffs_derivative_1 = [2, 6, 12]
        # y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
        coeffs_2 = [10, 2, 3, 4]
        coeffs_derivative_2 = [2, 6, 12]

        knots = [0, 1, 2]

        poly1 = Polynomial1d.polynomial_from_coefficients(coeffs_1)
        poly2 = Polynomial1d.polynomial_from_coefficients(coeffs_2)
        poly_derivative_1 = Polynomial1d.polynomial_from_coefficients(
            coeffs_derivative_1
        )
        poly_derivative_2 = Polynomial1d.polynomial_from_coefficients(
            coeffs_derivative_2
        )

        sp = Spline1d.spline_from_knots_and_polynomials(knots, [poly1, poly2])
        sp_de = Spline1d.spline_from_knots_and_polynomials(
            knots, [poly_derivative_1, poly_derivative_2]
        )
        self.assertEqual(sp.derivative_spline(1), sp_de)

    def test_spline_integral(self):
        # y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
        coeffs_1 = [1, 2, 3, 4]
        coeffs_integral_1 = [0, 1, 1, 1, 1]
        # y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
        coeffs_2 = [10, 2, 3, 4]
        coeffs_integral_2 = [0, 10, 1, 1, 1]

        knots = [0, 1, 2]

        poly1 = Polynomial1d.polynomial_from_coefficients(coeffs_1)
        poly2 = Polynomial1d.polynomial_from_coefficients(coeffs_2)
        poly_integral_1 = Polynomial1d.polynomial_from_coefficients(
            coeffs_integral_1
        )
        poly_integral_2 = Polynomial1d.polynomial_from_coefficients(
            coeffs_integral_2
        )

        sp = Spline1d.spline_from_knots_and_polynomials(knots, [poly1, poly2])
        sp_int = Spline1d.spline_from_knots_and_polynomials(
            knots, [poly_integral_1, poly_integral_2]
        )
        self.assertEqual(sp.integral_spline(1), sp_int)
