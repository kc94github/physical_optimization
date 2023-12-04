import numpy as np
import bisect

from typing import List
from src.abstract import Abstract
from src.geometry.polynomial1d import Polynomial1d


class Spline1d(Abstract):
    def __init__(self, knots: List[float], polynomials: List[Polynomial1d]):
        assert len(knots) == len(polynomials) + 1
        assert len(knots) >= 2
        self._knots = knots
        self._polynomials = polynomials

    def __repr__(self):
        s = "Cur Spline1d: \n"
        for i, poly in enumerate(self._polynomials):
            s += "number " + str(i) + ":" + poly.__repr__() + "\n"
        return s

    def __eq__(self, other) -> bool:
        if len(self) != len(other):
            return False
        if self.knots != other.knots:
            return False
        for poly1, poly2 in zip(self, other):
            if poly1 != poly2:
                return False
        return True

    @classmethod
    def spline_from_knots_and_polynomials(
        cls, knots: List[float], polynomials: List[Polynomial1d]
    ) -> "Spline1d":
        return cls(knots, polynomials)

    @property
    def knots(self) -> List[float]:
        return self._knots

    @property
    def polynomials(self) -> List[Polynomial1d]:
        return self._polynomials

    @property
    def order(self) -> int:
        return self._polynomials[0].order

    @property
    def total_param_number(self) -> int:
        if len(self) == 0:
            return 0
        return len(self) * self._polynomials[0].total_param_number

    def size(self):
        return len(self)

    def __len__(self):
        return len(self._polynomials)

    def __getitem__(self, index) -> Polynomial1d:
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
            return eval_func.__get__(self._polynomials[index], Polynomial1d)(
                relative_t, *args, **kwargs
            )

        return wrapper

    @spline_relative_eval
    def evaluate(self, t: float):
        return Polynomial1d.evaluate(self, t)

    @spline_relative_eval
    def derivative(self, t: float):
        return Polynomial1d.derivative(self, t)

    @spline_relative_eval
    def second_derivative(self, t: float):
        return Polynomial1d.second_derivative(self, t)

    @spline_relative_eval
    def third_derivative(self, t: float):
        return Polynomial1d.third_derivative(self, t)

    def derivative_spline(self, order: int = 1) -> "Spline1d":
        derivative_polys = []
        for poly in self._polynomials:
            derivative_polys.append(poly.derivative_polynomial(order))
        return Spline1d.spline_from_knots_and_polynomials(
            self.knots, derivative_polys
        )

    def integral_spline(self, order: int = 1) -> "Spline1d":
        integral_polys = []
        for poly in self._polynomials:
            integral_polys.append(poly.integral_polynomial(order))
        return Spline1d.spline_from_knots_and_polynomials(
            self.knots, integral_polys
        )
