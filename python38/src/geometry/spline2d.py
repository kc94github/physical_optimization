import numpy as np
import bisect

from typing import List
from src.abstract import Abstract
from src.geometry.polynomial1d import Polynomial1d
from src.geometry.polynomial2d import Polynomial2d
from src.geometry.spline1d import Spline1d


class Spline2d(Abstract):
    def __init__(self, spline_x: Spline1d, spline_y: Spline1d):
        assert len(spline_x) == len(spline_y)
        assert spline_x.knots == spline_y.knots
        assert len(spline_x.knots) >= 2
        assert spline_x.order == spline_y.order

        self._knots = spline_x.knots
        self._x_spline = spline_x
        self._y_spline = spline_y

    def __repr__(self):
        s = "Cur Spline2d: \n"
        s += "  X spline: " + self._x_spline.__repr__() + "\n"
        s += "  Y spline: " + self._y_spline.__repr__() + "\n"
        return s

    def __eq__(self, other) -> bool:
        return (
            self.x_spline == other.x_spline and self.y_spline == other.y_spline
        )

    @classmethod
    def from_xy_splines(
        cls, x_spline: Spline1d, y_spline: Spline1d
    ) -> "Spline2d":
        return cls(x_spline, y_spline)

    @classmethod
    def from_xy_knots_and_coefficients(
        cls,
        knots: List[float],
        x_coefficients: List[List[float]],
        y_coefficients: List[List[float]],
    ) -> "Spline2d":
        assert len(x_coefficients) == len(y_coefficients)

        x_poly: List[Polynomial1d] = []
        y_poly: List[Polynomial1d] = []
        var_size = -1
        for x_coeffs, y_coeffs in zip(x_coefficients, y_coefficients):
            assert len(x_coeffs) == len(y_coeffs)
            if var_size == -1:
                var_size = len(x_coeffs)
            else:
                assert len(x_coeffs) == var_size
            x_poly.append(Polynomial1d.polynomial_from_coefficients(x_coeffs))
            y_poly.append(Polynomial1d.polynomial_from_coefficients(y_coeffs))

        sp_x = Spline1d.spline_from_knots_and_polynomials(knots, x_poly)
        sp_y = Spline1d.spline_from_knots_and_polynomials(knots, y_poly)
        return Spline2d.from_xy_splines(sp_x, sp_y)

    def size(self) -> int:
        return len(self)

    def __len__(self):
        return len(self._x_spline)

    def size(self):
        return len(self)

    def __getitem__(self, index) -> Polynomial2d:
        return self.line_segment(index)

    def line_segment(self, index) -> Polynomial2d:
        return Polynomial2d.polynomial2d_from_polynomial1d(
            self.x_polynomials[index], self.y_polynomials[index]
        )

    @property
    def knots(self) -> List[float]:
        return self._knots

    def knot_segment(self, index) -> List[float]:
        return self._knots[index : index + 2]

    @property
    def x_spline(self) -> Spline1d:
        return self._x_spline

    @property
    def x_polynomials(self) -> List[Polynomial1d]:
        return self._x_spline.polynomials

    @property
    def y_spline(self) -> Spline1d:
        return self._y_spline

    @property
    def y_polynomials(self) -> List[Polynomial1d]:
        return self._y_spline.polynomials

    @property
    def order(self) -> int:
        return self._x_spline.order

    @property
    def total_param_number(self) -> int:
        return (
            self._x_spline.total_param_number
            + self._y_spline.total_param_number
        )

    def evaluate_x(self, t: float) -> float:
        return self._x_spline.evaluate(t)

    def evaluate_y(self, t: float) -> float:
        return self._y_spline.evaluate(t)

    def evaluate(self, t: float) -> List[float]:
        return [self.evaluate_x(t), self.evaluate_y(t)]

    def derivative_x(self, t: float) -> float:
        return self._x_spline.derivative(t)

    def derivative_y(self, t: float) -> float:
        return self._y_spline.derivative(t)

    def derivative(self, t: float) -> List[float]:
        return [self.derivative_x(t), self.derivative_y(t)]

    def second_derivative_x(self, t: float) -> float:
        return self._x_spline.second_derivative(t)

    def second_derivative_y(self, t: float) -> float:
        return self._y_spline.second_derivative(t)

    def second_derivative(self, t: float) -> List[float]:
        return [self.second_derivative_x(t), self.second_derivative_y(t)]

    def third_derivative_x(self, t: float) -> float:
        return self._x_spline.third_derivative(t)

    def third_derivative_y(self, t: float) -> float:
        return self._y_spline.third_derivative(t)

    def third_derivative(self, t: float) -> List[float]:
        return [self.third_derivative_x(t), self.third_derivative_y(t)]

    def derivative_spline(self, order: int = 1) -> "Spline2d":
        x_derivative_spline = self._x_spline.derivative_spline(order)
        y_derivative_spline = self._y_spline.derivative_spline(order)
        return Spline2d.from_xy_splines(
            x_derivative_spline, y_derivative_spline
        )

    def integral_spline(self, order: int = 1) -> "Spline2d":
        x_integral_spline = self._x_spline.integral_spline(order)
        y_integral_spline = self._y_spline.integral_spline(order)
        return Spline2d.from_xy_splines(x_integral_spline, y_integral_spline)
