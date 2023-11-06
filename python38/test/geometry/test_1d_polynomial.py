import unittest
import numpy as np
from src.geometry.polynomial1d import Polynomial1d


class TestPolynomial1d(unittest.TestCase):
    def test_poly_basic(self):
        # y = x^3 + 2x^2+ 3x + 4
        coeffs = [4, 3, 2, 1]
        poly = Polynomial1d.polynomial_from_coefficients(coeffs)
        print(poly)

        self.assertEqual(poly.order, 3)
        self.assertEqual(poly.total_param_number, 4)
        self.assertEqual(poly.coefficients, coeffs)

    def test_evaluations(self):
        # y = x^3 + 2x^2+ 3x + 4
        coeffs = [4, 3, 2, 1]
        poly = Polynomial1d.polynomial_from_coefficients(coeffs)

        self.assertEqual(poly.evaluate(2.0), 26)
        self.assertEqual(poly.derivative(2.0), 23)
        self.assertEqual(poly.second_derivative(2.0), 16)
        self.assertEqual(poly.third_derivative(2.0), 6)

    def test_derivative_poly(self):
        # y = x^3 + 2x^2+ 3x + 4
        coeffs = [4, 3, 2, 1]
        first_order_coeffs = [3, 4, 3]
        second_order_coeffs = [4, 6]
        third_order_coeffs = [6]
        poly = Polynomial1d.polynomial_from_coefficients(coeffs)

        first_order_poly = poly.derivative_polynomial()
        self.assertEqual(first_order_poly.order, 2)
        self.assertEqual(first_order_poly.total_param_number, 3)
        self.assertEqual(first_order_poly.coefficients, first_order_coeffs)

        second_order_poly = first_order_poly.derivative_polynomial()
        self.assertEqual(second_order_poly.order, 1)
        self.assertEqual(second_order_poly.total_param_number, 2)
        self.assertEqual(second_order_poly.coefficients, second_order_coeffs)

        third_order_poly = second_order_poly.derivative_polynomial()
        self.assertEqual(third_order_poly.order, 0)
        self.assertEqual(third_order_poly.total_param_number, 1)
        self.assertEqual(third_order_poly.coefficients, third_order_coeffs)

        self.assertEqual(third_order_poly, poly.derivative_polynomial(3))

    def test_integral_poly(self):
        # y = x^3 + 3x^2+ 3x + 4
        coeffs = [4, 3, 3, 1]
        first_order_coeffs = [0, 4, 3 / 2, 3 / 3, 1 / 4]
        second_order_coeffs = [0, 0, 2, 0.5, 0.25, 0.05]
        poly = Polynomial1d.polynomial_from_coefficients(coeffs)

        first_order_poly = poly.integral_polynomial()
        self.assertEqual(first_order_poly.order, 4)
        self.assertEqual(first_order_poly.total_param_number, 5)
        self.assertEqual(first_order_poly.coefficients, first_order_coeffs)

        second_order_poly = first_order_poly.integral_polynomial()
        self.assertEqual(second_order_poly.order, 5)
        self.assertEqual(second_order_poly.total_param_number, 6)
        self.assertEqual(second_order_poly.coefficients, second_order_coeffs)

        self.assertEqual(second_order_poly, poly.integral_polynomial(2))
