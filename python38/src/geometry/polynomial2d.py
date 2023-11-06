import numpy as np
from typing import List
from src.abstract import Abstract
from src.geometry.polynomial1d import Polynomial1d


class Polynomial2d(Abstract):
    def __init__(
        self, x_coefficients: List[float], y_coefficients: List[float]
    ):
        assert len(x_coefficients) == len(y_coefficients)
        self._order = len(x_coefficients) - 1
        self._total_param = len(x_coefficients) * 2
        self._x_polynomial = Polynomial1d(x_coefficients)
        self._y_polynomial = Polynomial1d(y_coefficients)

    @classmethod
    def polynomial_from_coefficients(
        cls, x_coefficients: List[float], y_coefficients: List[float]
    ):
        return cls(x_coefficients, y_coefficients)

    @classmethod
    def polynomial2d_from_polynomial1d(
        cls, x_polynomial: Polynomial1d, y_polynomial: Polynomial1d
    ):
        return cls(x_polynomial.coefficients, y_polynomial.coefficients)

    def __repr__(self) -> str:
        return f"Polynomial2D with order: {self._order}, \
            x_coeff: {self._x_polynomial.coefficients}, \
            y_coeff: {self._y_polynomial.coefficients}"

    def __eq__(self, other) -> bool:
        return (
            self.x_polynomial == other.x_polynomial
            and self.y_polynomial == other.y_polynomial
        )

    @property
    def x_coefficients(self) -> List[float]:
        return self._x_polynomial.coefficients

    @property
    def y_coefficients(self) -> List[float]:
        return self._y_polynomial.coefficients

    @property
    def x_polynomial(self) -> Polynomial1d:
        return self._x_polynomial

    @property
    def y_polynomial(self) -> Polynomial1d:
        return self._y_polynomial

    @property
    def order(self) -> int:
        return self._order

    @property
    def total_param_number(self) -> int:
        return self._total_param

    def evaluate_x(self, t: float) -> float:
        return self.x_polynomial.evaluate(t)

    def evaluate_y(self, t: float) -> float:
        return self.y_polynomial.evaluate(t)

    def evaluate(self, t: float) -> List[float]:
        return [self.evaluate_x(t), self.evaluate_y(t)]

    def derivative_x(self, t: float) -> float:
        return self.x_polynomial.derivative(t)

    def derivative_y(self, t: float) -> float:
        return self.y_polynomial.derivative(t)

    def derivative(self, t: float) -> List[float]:
        return [self.derivative_x(t), self.derivative_y(t)]

    def second_derivative_x(self, t: float) -> float:
        return self.x_polynomial.second_derivative(t)

    def second_derivative_y(self, t: float) -> float:
        return self.y_polynomial.second_derivative(t)

    def second_derivative(self, t: float) -> List[float]:
        return [
            self.second_derivative_x(t),
            self.second_derivative_y(t),
        ]

    def third_derivative_x(self, t: float) -> float:
        return self.x_polynomial.third_derivative(t)

    def third_derivative_y(self, t: float) -> float:
        return self.y_polynomial.third_derivative(t)

    def third_derivative(self, t: float) -> List[float]:
        return [self.third_derivative_x(t), self.third_derivative_y(t)]

    def derivative_polynomial(self, order: int = 1) -> "Polynomial2d":
        x_coeffs = self._x_polynomial.derivative_polynomial(order).coefficients
        y_coeffs = self._y_polynomial.derivative_polynomial(order).coefficients
        return Polynomial2d.polynomial_from_coefficients(x_coeffs, y_coeffs)

    def integral_polynomial(self, order: int = 1) -> "Polynomial2d":
        x_coeffs = self._x_polynomial.integral_polynomial(order).coefficients
        y_coeffs = self._y_polynomial.integral_polynomial(order).coefficients
        return Polynomial2d.polynomial_from_coefficients(x_coeffs, y_coeffs)
