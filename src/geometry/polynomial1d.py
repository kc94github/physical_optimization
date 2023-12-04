import numpy as np
from typing import List
from src.abstract import Abstract
from src.geometry.coefficient_base import CoefficientBase


class Polynomial1d(CoefficientBase):
    def __init__(self, coefficients: List[float]):
        self._order = len(coefficients) - 1
        self._total_param = len(coefficients)
        self._coefficients = coefficients
        CoefficientBase.__init__(self, self._order)

    @classmethod
    def polynomial_from_coefficients(cls, coefficients: List[float]):
        return cls(coefficients)

    def __repr__(self) -> str:
        return f"Polynomial1d with order {self._order} and ascending coefficients {self._coefficients}. "

    def summation(coefficient_func):
        def wrapper(self, t: float, *args, **kwargs):
            summation = 0.0
            res = coefficient_func(self, t)
            for param, coeff in zip(self._coefficients, res):
                summation += param * coeff
            return summation

        return wrapper

    def __eq__(self, other) -> bool:
        return self.coefficients == other.coefficients

    @property
    def order(self) -> int:
        return self._order

    @property
    def total_param_number(self) -> int:
        return self._total_param

    @property
    def coefficients(self):
        return self._coefficients

    @summation
    def evaluate(self, t: float) -> float:
        return self.t_coefficient(t)

    @summation
    def derivative(self, t: float) -> float:
        return self.t_first_derivative_coefficient(t)

    @summation
    def second_derivative(self, t: float) -> float:
        return self.t_second_derivative_coefficient(t)

    @summation
    def third_derivative(self, t: float) -> float:
        return self.t_third_derivative_coefficient(t)

    def derivative_polynomial(self, order: int = 1) -> "Polynomial1d":
        old_coefficients = self._coefficients
        for o in range(order):
            new_coefficients = []
            for i in range(1, len(old_coefficients)):
                new_coefficients.append(old_coefficients[i] * i)
            old_coefficients = new_coefficients
        return Polynomial1d.polynomial_from_coefficients(old_coefficients)

    def integral_polynomial(self, order: int = 1) -> "Polynomial1d":
        old_coefficients = self._coefficients
        for o in range(order):
            new_coefficients = [0]
            for i in range(len(old_coefficients)):
                new_coefficients.append(old_coefficients[i] * 1.0 / (i + 1))
            old_coefficients = new_coefficients
        return Polynomial1d.polynomial_from_coefficients(old_coefficients)
