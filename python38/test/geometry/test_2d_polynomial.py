import unittest
import numpy as np
from src.geometry.polynomial import Polynomial
from src.geometry.polynomial2d import Polynomial2d


class TestPolynomial2d(unittest.TestCase):
    def test_poly_basic(self):
        # x = t^3 + 2t^2+ 3t + 4
        # y = 2t^3 + 3
        x_coeffs = [4, 3, 2, 1]
        y_coeffs = [3, 0, 0, 2]
        poly = Polynomial2d.polynomial_from_coefficients(x_coeffs, y_coeffs)
        print(poly)

        self.assertEqual(poly.order, 3)
        self.assertEqual(poly.total_param_number, 8)
        self.assertEqual(poly.x_coefficients, x_coeffs)
        self.assertEqual(poly.y_coefficients, y_coeffs)

    def test_poly_error(self):
        # x = t^3 + 2t^2+ 3t + 4
        # y = 2t^3 + 3
        x_coeffs = [4, 3, 2, 1]
        y_coeffs = [0, 0, 2]
        with self.assertRaises(AssertionError):
            poly = Polynomial2d.polynomial_from_coefficients(
                x_coeffs, y_coeffs
            )

    def test_poly_evals(self):
        # x = t^3 + 2t^2+ 3t + 4
        # y = 2t^3 + 3
        x_coeffs = [4, 3, 2, 1]
        y_coeffs = [3, 0, 0, 2]
        poly = Polynomial2d.polynomial_from_coefficients(x_coeffs, y_coeffs)

        self.assertEqual(poly.evaluate(2.0), [26.0, 19.0])
        self.assertEqual(poly.derivative(2.0), [23.0, 24.0])
        self.assertEqual(poly.second_derivative(2.0), [16.0, 24.0])
        self.assertEqual(poly.third_derivative(2.0), [6.0, 12.0])

    def test_derivative_poly(self):
        # x = t^3 + 2t^2+ 3t + 4
        # y = 2t^3 + 3
        x_coeffs = [4, 3, 2, 1]
        y_coeffs = [3, 0, 0, 2]

        x_dot_coeffs = [3, 4, 3]
        y_dot_coeffs = [0, 0, 6]

        x_double_dot_coeffs = [4, 6]
        y_double_dot_coeffs = [0, 12]

        poly = Polynomial2d.polynomial_from_coefficients(x_coeffs, y_coeffs)

        self.assertEqual(
            poly.derivative_polynomial(1),
            Polynomial2d.polynomial_from_coefficients(
                x_dot_coeffs, y_dot_coeffs
            ),
        )
        self.assertEqual(
            poly.derivative_polynomial(2),
            Polynomial2d.polynomial_from_coefficients(
                x_double_dot_coeffs, y_double_dot_coeffs
            ),
        )

    def test_integral_poly(self):
        # x = t^3 + 3t^2+ 3t + 4
        # y = 2t^3 + 3
        x_coeffs = [4, 3, 3, 1]
        y_coeffs = [3, 0, 0, 2]

        x_integral_coeffs = [0, 4, 1.5, 1, 0.25]
        y_integral_coeffs = [0, 3, 0, 0, 0.5]

        x_double_integral_coeffs = [0, 0, 2, 0.5, 0.25, 0.05]
        y_double_integral_coeffs = [0, 0, 1.5, 0, 0, 0.1]

        poly = Polynomial2d.polynomial_from_coefficients(x_coeffs, y_coeffs)

        self.assertEqual(
            poly.integral_polynomial(1),
            Polynomial2d.polynomial_from_coefficients(
                x_integral_coeffs, y_integral_coeffs
            ),
        )
        self.assertEqual(
            poly.integral_polynomial(2),
            Polynomial2d.polynomial_from_coefficients(
                x_double_integral_coeffs, y_double_integral_coeffs
            ),
        )
