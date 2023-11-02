import numpy as np
from abstract import Abstract

class CoefficientBase(Abstract):

    def __init__(self, order):
        self._order = order

    @property
    def order(self) -> int:
        return self._order

    def t_coefficient(self, t):
        res = np.zeros(self._order + 1)
        res[0] = 1
        for i in range(1, res.shape[0]):
            res[i] = res[i-1] * t
        return res

    def t_first_derivative_coefficient(self, t):
        res = np.zeros(self._order + 1)
        t_coeff = self.t_coefficient(t)
        for i in range(1, res.shape[0]):
            res[i] = t_coeff[i-1] * i
        return res

    def t_second_derivative_coefficient(self, t):
        res = np.zeros(self._order + 1)
        t_first_derivative_coeff = self.t_first_derivative_coefficient(
            t)
        for i in range(2, res.shape[0]):
            res[i] = t_first_derivative_coeff[i-1] * i
        return res

    def t_third_derivative_coefficient(self, t):
        res = np.zeros(self._order + 1)
        t_second_derivative_coeff = self.t_second_derivative_coefficient(
            t)
        for i in range(3, res.shape[0]):
            res[i] = t_second_derivative_coeff[i-1] * i
        return res