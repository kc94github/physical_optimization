import unittest
import numpy as np
from src.geometry.bezier1d import Bezier1d


class TestBezierBase(unittest.TestCase):
    def test_bezier_1d_basic(self):
        pt_coefficients = [1, 2, 3, 4]
        bezier = Bezier1d.bezier_from_coefficients(pt_coefficients)

        self.assertEqual(bezier.order, 3)
        self.assertEqual(bezier.total_param_number, 4)

        print(bezier.form)
