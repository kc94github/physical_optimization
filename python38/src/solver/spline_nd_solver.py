import numpy as np
import bisect
from typing import List, Tuple

from src.abstract import Abstract
from src.solver.solver_impl import SolverImpl
from src.geometry.coefficient_base import CoefficientBase


class SplineNdSolver(CoefficientBase):
    def __init__(
        self, knots: List[float], spline_order: int, dimension: int = 1
    ):
        assert len(knots) >= 2
        self._knots = knots
        self._spline_order = spline_order
        self._dimension = dimension

        self._total_param_length = (
            self._dimension * (self._spline_order + 1) * (len(self._knots) - 1)
        )
        self._solver = SolverImpl(self._total_param_length)
        CoefficientBase.__init__(self, order=self._spline_order)

    def __repr__(self):
        return f"Spline Nd Solver with: \n knots:{self._knots} \n spline_order:{self._spline_order} \n dimension:{self._dimension}"

    @property
    def solver(self):
        return self._solver

    def solve(self, solver_name: str = "osqp", solution_only: bool = True):
        return self._solver.solve(solver_name, solution_only)

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

    def get_knot_index_and_coefficient(coefficient_derivative_func):
        def wrapper(self, t: float, *args, **kwargs):
            assert (
                kwargs == {}
            ), "get_knot_index_and_coefficient error, do not use named input variable!"
            index = self._search_prev_knot_index(t=t)
            relative_time = t - self._knots[index]
            return index, coefficient_derivative_func(
                self, relative_time, *args
            )

        return wrapper

    def apply_equality_constraint(knot_index_and_coefficient):
        def wrapper(self, t: float, *args, **kwargs):
            assert (
                kwargs == {}
            ), "apply_equality_constraint error, do not use named input variable!"
            dim = self._dimension
            # Get knot index that the input t belongs to, and
            # Get coefficient that the order of derivative that function knot_index_and_coefficient has specified
            index, coeff = knot_index_and_coefficient(self, t, *args)

            # Apply equality constraint Ax = b, A is of size dim * total_param_length, and total_param_length includes all knots;
            # And we only get constraint values on the target knot index
            matrix_A = np.zeros([dim, self._total_param_length])
            start_index = index * (dim * (self._spline_order + 1))
            for i in range(dim):
                matrix_A[
                    i, start_index : start_index + (self._spline_order + 1)
                ] = coeff
                start_index += self._spline_order + 1

            assert (
                len(*args) == dim
            ), "apply_equality_constraint error, input derivative point size not matching dimension size"
            matrix_B = np.array([*args])
            self._solver.add_equality_constraint(matrix_A, matrix_B)

        return wrapper

    def apply_inequality_constraint(knot_index_and_coefficient):
        def wrapper(self, t: float, *args, **kwargs):
            assert (
                kwargs == {}
            ), "apply_inequality_constraint error, do not use named input variable!"
            dim = self._dimension
            # Get knot index that the input t belongs to, and
            # Get coefficient that the order of derivative that function knot_index_and_coefficient has specified
            index, coeff = knot_index_and_coefficient(t, *args)

            # Apply equality constraint Ax = b, A is of size dim * total_param_length, and total_param_length includes all knots;
            # And we only get constraint values on the target knot index
            matrix_A = np.zeros([dim, self._total_param_length])
            start_index = index * (dim * (self._spline_order + 1))
            for i in range(dim):
                matrix_A[
                    i, start_index : start_index + (self._spline_order + 1)
                ] = coeff
                start_index += self._spline_order + 1

            assert (
                len(*args) == dim
            ), "apply_inequality_constraint error, input derivative point size not matching dimension size"
            matrix_B = np.array([*args])
            self._solver.add_inequality_constraint(matrix_A, matrix_B)

        return wrapper

    @apply_equality_constraint
    @get_knot_index_and_coefficient
    def add_point_constraint(self, t: float, *args, **kwargs) -> bool:
        return self.t_coefficient(t)

    @apply_equality_constraint
    @get_knot_index_and_coefficient
    def add_point_first_derivative_constraint(self, t: float, *args, **kwargs):
        return self.t_first_derivative_coefficient(t)

    @apply_equality_constraint
    @get_knot_index_and_coefficient
    def add_point_second_derivative_constraint(
        self, t: float, *args, **kwargs
    ):
        return self.t_second_derivative_coefficient(t)

    @apply_equality_constraint
    @get_knot_index_and_coefficient
    def add_point_third_derivative_constraint(self, t: float, *args, **kwargs):
        return self.t_third_derivative_coefficient(t)
