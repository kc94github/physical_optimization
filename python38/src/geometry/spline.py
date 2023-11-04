import numpy as np
import bisect

from typing import List
from src.abstract import Abstract
from src.geometry.polynomial import Polynomial


class Spline(Abstract):
    def __init__(self, knots: List[float], polynomials: List[Polynomial]):
        assert len(knots) == len(polynomials) + 1
        self._knots = knots
        self._polynomials = polynomials

    def __repr__(self):
        s = "Cur Spline: \n"
        for i, poly in enumerate(self._polynomials):
            s += "number " + str(i) + ":" + poly.__repr__() + "\n"
        return s

    @classmethod
    def spline_from_knots_and_polynomials(
        cls, knots: List[float], polynomials: List[Polynomial]
    ) -> "Spline":
        return cls(knots, polynomials)

    @property
    def knots(self):
        return self._knots

    @property
    def polynomials(self):
        return self._polynomials

    def size(self):
        return len(self)

    def __len__(self):
        return len(self._polynomials)

    def __getitem__(self, index):
        return self._polynomials[index]

    def knot_segment(self, index):
        return self.knots[index : index + 2]

    def line_segment(self, index):
        return self.polynomials[index]

    def _search_prev_knot_index(self, t: float):
        i = bisect.bisect_right(self._knots, t)
        if i == len(self._knots):
            i -= 1
        if i > 0:
            return i - 1
        else:
            raise Exception("Find index failed with t < t_knots[0]")

    def search_prev_knot_index(self, t: float):
        return self._search_prev_knot_index(t)

    def spline_relative_eval(eval_func):
        def wrapper(self, t: float, *args, **kwargs):
            index = self._search_prev_knot_index(t)
            relative_t = t - self._knots[index]
            return eval_func.__get__(self._polynomials[index], Polynomial)(
                relative_t, *args, **kwargs
            )

        return wrapper

    @spline_relative_eval
    def evaluate(self, t: float):
        return Polynomial.evaluate(self, t)

    @spline_relative_eval
    def derivative(self, t: float):
        return Polynomial.derivative(self, t)

    @spline_relative_eval
    def second_derivative(self, t: float):
        return Polynomial.second_derivative(self, t)

    @spline_relative_eval
    def third_derivative(self, t: float):
        return Polynomial.third_derivative(self, t)
