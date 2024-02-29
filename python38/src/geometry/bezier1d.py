import numpy as np
from typing import List
from src.abstract import Abstract
from src.geometry.bezier_base import BezierBase, pascal_triangle


class Bezier1d(BezierBase):
    def __init__(self, coefficients: List[float]):
        self._order = len(coefficients) - 1
        self._total_param = len(coefficients)
        self._coefficients = coefficients
        BezierBase.__init__(self, self._order)

    @classmethod
    def bezier_from_coefficients(cls, coefficients: List[float]):
        return cls(coefficients)

    def __repr__(self) -> str:
        return f"Bezier 1d with order: {self._order}, with coefficients: {self._order}"

    def summation(coefficient_func):
        def wrapper(self, t: float, *args, **kwargs):
            summation = 0.0
            res = coefficient_func(self, t)
            for param, coeff in zip(self._coefficients, res):
                summation += param * coeff
            return summation

        return wrapper

    @property
    def form(self) -> str:
        multiply_coefficients = pascal_triangle()[self._order]
        result = "Bezier form is: \n"
        for i in range(self._total_param):
            result += (
                str(multiply_coefficients[i])
                + "*(1-t)^"
                + str(self._order - i)
                + "*t^"
                + str(i)
                + "* "
                + str(self._coefficients[i])
            )
            if i == self._total_param - 1:
                return result
            result += " + "
        return result

    @property
    def order(self) -> int:
        return self._order

    @property
    def total_param_number(self) -> int:
        return self._total_param

    @property
    def coefficients(self) -> List[float]:
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

    def derivative_bezier(self, order: int = 1) -> "Bezier1d":
        assert order >= 1 and order <= 3
        assert (
            self._order - order >= 1
        )  # Derivative bezier will need to be > 1 order
        old_coefficients = self._coefficients
        new_coefficients = []
        if order == 1:
            multiply_coeff = self._order
            for i in range(1, len(old_coefficients)):
                new_coefficients.append(
                    multiply_coeff
                    * (old_coefficients[i] - old_coefficients[i - 1])
                )
        elif order == 2:
            multiply_coeff = self._order * (self._order - 1)
            for i in range(2, len(old_coefficients)):
                new_coefficients.append(
                    multiply_coeff
                    * (
                        old_coefficients[i]
                        - 2.0 * old_coefficients[i - 1]
                        + old_coefficients[i - 2]
                    )
                )
        elif order == 3:
            multiply_coeff = (
                self._order * (self._order - 1) * (self._order - 2)
            )
            for i in range(3, len(old_coefficients)):
                new_coefficients.append(
                    multiply_coeff
                    * (
                        old_coefficients[i]
                        - 3.0 * old_coefficients[i - 1]
                        + 3.0 * old_coefficients[i - 2]
                        - old_coefficients[i - 3]
                    )
                )
        return Bezier1d.bezier_from_coefficients(new_coefficients)
