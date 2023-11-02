import numpy as np
from abstract import Abstract
from geometry.coefficient_base import CoefficientBase

class Polynomial(Abstract, CoefficientBase):

    def __init__(self, coefficients: List[float]):
        self._order = len(coefficients) - 1
        self._total_param = len(coefficients)
        self._coefficients = coefficients
        CoefficientBase.__init__(self._order)

    def __repr__(self) -> str:
        return f"Polynomial with order {self._order} and coefficients {self._coefficients}. "

    def summation(coefficient_func):
        def wrapper(self, t: float, *args, **kwargs):
            summation = 0.0
            res = coefficient_func(self, t)
            for param, coeff in zip(self._parameters, res):
                summation += param * coeff
            return summation
        return wrapper

    @property
    def order(self) -> int:
        return self._order

    @property
    def total_number(self) -> int:
        return self._total_param

    @property
    def coefficients(self):
        return self._coefficients

    @summation
    def evaluate(self, t: float) -> float:
        return self.t_coefficient(t)

    def derivative_evaluate(self, t: float) -> float:
        return self.t_first_derivative_coefficient(t)

    def second_derivative_evaluate(self, t: float) -> float:
        return self.t_second_derivative_coefficient(t)

    def third_derivative_evaluate(self, t: float) -> float:
        return self.t_third_derivative_coefficient(t)

    
