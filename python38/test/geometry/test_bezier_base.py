import unittest
import numpy as np
from src.geometry.bezier_base import BezierBase


class TestBezierBase(unittest.TestCase):
    def test_bezier_base_basic(self):
        bezier = BezierBase(3)

        self.assertEqual(bezier.order, 3)
        self.assertEqual(
            bezier.t_coefficient(0.5), [0.125, 0.375, 0.375, 0.125]
        )
        self.assertEqual(
            bezier.t_first_derivative_coefficient(0.5), [0.75, 1.5, 0.75]
        )
        self.assertEqual(
            bezier.t_second_derivative_coefficient(0.5), [3.0, 3.0]
        )
        with self.assertRaises(Exception):
            bezier.t_third_derivative_coefficient(0.5)

        self.assertEqual(bezier.order, 3)
        self.assertEqual(bezier.t_coefficient(1.0), [0, 0, 0, 1])
        self.assertEqual(bezier.t_first_derivative_coefficient(1.0), [0, 0, 3])
        self.assertEqual(bezier.t_second_derivative_coefficient(1.0), [0, 6])
        with self.assertRaises(Exception):
            bezier.t_third_derivative_coefficient(1.0)
