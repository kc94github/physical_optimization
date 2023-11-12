import numpy as np
import bisect

from src.abstract import Abstract
from src.solver.solver_impl import SolverImpl


class SplineNdSolver(Abstract):
    def __init__(
        self, knots: List[float], spline_order: int, dimension: int = 1
    ):
        assert len(knots) >= 2
        self._knots = knots
        self._spline_order = spline_order
        self._dimension = dimension

        self._total_param_length = (
            self._dimension * (self._spline_order + 1) * (self._knots - 1)
        )
        self._solver = SolverImpl(self._total_param_length)

    def __repr__(self):
        return f"Spline Nd Solver with: \n knots:{self._knots} \n spline_order:{self._spline_order} \n dimension:{self._dimension}"

    @property
    def knots(self):
        return self._knots

    @property
    def spline_order(self):
        return self._spline_order

    @property
    def dimension(self):
        return self._dimension

    @property
    def total_param_length(self):
        return self._total_param_length

    def __len__(self):
        return self.total_param_length

    def _search_prev_knot_index(self, t: float):
        i = bisect.bisect_right(self._knots, t)
        if i == len(self._knots):
            i -= 1
        if i > 0:
            return i - 1
        else:
            raise Exception("Find index failed with t < t_knots[0]")

    def add_regularization(self, regularization_param: float) -> bool:
        regularization_matrix = (
            np.identity(self._total_param_length) * regularization_param
        )
        return self._solver.add_to_objective_function(
            0,
            0,
            self._total_param_length,
            self._total_param_length,
            regularization_matrix,
        )
