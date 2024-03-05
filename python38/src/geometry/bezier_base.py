import numpy as np
from src.abstract import Abstract


def pascal_triangle():
    return [
        [1.0],
        [1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 3.0, 1.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        [1.0, 7.0, 21.0, 35.0, 35.0, 21.0, 7.0, 1.0],
        [1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0],
        [1.0, 9.0, 36.0, 84.0, 126.0, 126.0, 84.0, 36.0, 9.0, 1.0],
    ]


class BezierBase(Abstract):
    def __init__(self, order):
        assert order >= 1 and order <= 9
        self._order = order

    def __repr__(self):
        return f"BezierBase with order: {self._order}"

    @property
    def order(self) -> int:
        return self._order

    @staticmethod
    def _get_t_coefficient(order: int, t: float):
        assert t >= 0 and t <= 1
        assert order >= 1 and order <= 9
        res = pascal_triangle()[order]  # Size of self._order + 1
        one_minus_t = 1.0 - t
        first = [1.0]
        second = [1.0]
        for i in range(order):
            first.append(first[-1] * one_minus_t)
            second.append(second[-1] * t)

        print("First:", first)
        print("Second:", second)
        for i in range(order + 1):
            print("Co:", (first[order - i] * second[i]))
            res[i] *= first[order - i] * second[i]
        print("Num: ", res)
        return res

    def t_coefficient(self, t: float):
        return BezierBase._get_t_coefficient(self._order, t)

    def t_first_derivative_coefficient(self, t: float):
        assert self._order >= 2
        coeffs = BezierBase._get_t_coefficient(self._order - 1, t)
        multiply_coeff = self._order
        coeffs = [element * multiply_coeff for element in coeffs]
        return coeffs

    def t_second_derivative_coefficient(self, t):
        assert self._order >= 3
        coeffs = BezierBase._get_t_coefficient(self._order - 2, t)
        multiply_coeff = self._order * (self._order - 1)
        coeffs = [element * multiply_coeff for element in coeffs]
        return coeffs

    def t_third_derivative_coefficient(self, t):
        assert self._order >= 4
        coeffs = BezierBase._get_t_coefficient(self._order - 3, t)
        multiply_coeff = self._order * (self._order - 1) * (self._order - 2)
        coeffs = [element * multiply_coeff for element in coeffs]
        return coeffs
