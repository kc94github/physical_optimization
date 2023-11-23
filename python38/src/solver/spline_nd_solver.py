import numpy as np
import bisect
from typing import List, Tuple

from src.abstract import Abstract
from src.solver.solver_impl import SolverImpl
from src.geometry.coefficient_base import CoefficientBase


class SplineNdSolver(CoefficientBase, Abstract):
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

    def hessian_matrix(self):
        return self._solver.hessian_matrix

    def gradient_vector(self):
        return self._solver.gradient_vector

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
            for arg in args:
                assert(isinstance(arg, (int, float))), "get argument not in float or int!"
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
            for arg in args:
                assert(isinstance(arg, (int, float))), "get argument not in float or int!"
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
                len([*args]) == dim
            ), "apply_equality_constraint error, input derivative point size not matching dimension size"
            matrix_B = np.array([*args])
            self._solver.add_equality_constraint(matrix_A, matrix_B)

        return wrapper

    def apply_inequality_constraint(knot_index_and_coefficient):
        def wrapper(self, t: float, *args, **kwargs):
            assert (
                kwargs == {}
            ), "apply_inequality_constraint error, do not use named input variable!"
            for arg in args:
                assert(isinstance(arg, (int, float))), "get argument not in float or int!"
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
                len([*args]) == dim
            ), "apply_inequality_constraint error, input derivative point size not matching dimension size"
            matrix_B = np.array([*args])
            self._solver.add_inequality_constraint(matrix_A, matrix_B)

        return wrapper

    def add_smooth_constraint(self, order=0) -> bool:
        assert (
            order <= 3
        ), "Smoothing only support up to order = 3, please check"
        rst = self._add_smooth_constraint_order_zero()
        if order == 0:
            return rst
        rst &= self._add_smooth_constraint_order_one()
        if order == 1:
            return rst
        rst &= self._add_smooth_constraint_order_two()
        if order == 2:
            return rst
        rst &= self._add_smooth_constraint_order_three()
        if order >= 3:
            return rst
        return False  # Should not reach here

    def _add_smooth_constraint_order_zero(self) -> bool:
        assert (
            len(self._knots)
        ) >= 3, "_add_smooth_constraint_order_zero error, knot size < 3"
        dim = self._dimension
        smooth_matrix_A = np.zeros(
            [dim * (len(self._knots) - 2), self._total_param_length]
        )
        smooth_matrix_B = np.zeros(dim * (len(self._knots) - 2))
        for i in range(len(self._knots) - 2):
            # Current connection knot
            # Since we are using relative_t for a knot, we need the relative_t to use the coefficient function as well
            relative_t = self._knots[i + 1] - self._knots[i]
            coeff = self.t_coefficient(relative_t)
            index_offset = i * dim * (self._spline_order + 1)
            next_index_offset = (i + 1) * dim * (self._spline_order + 1)
            for n in range(dim):
                for j in range(self._spline_order + 1):
                    smooth_matrix_A[i * dim + n][
                        index_offset + n * (self._spline_order + 1) + j
                    ] = coeff[j]
                smooth_matrix_A[i * dim + n][
                    next_index_offset + n * (self._spline_order + 1)
                ] = -1
        return self._solver.add_equality_constraint(
            smooth_matrix_A, smooth_matrix_B
        )

    def _add_smooth_constraint_order_one(self) -> bool:
        assert (
            len(self._knots)
        ) >= 3, "_add_smooth_constraint_order_one error, knot size < 3"
        dim = self._dimension
        smooth_matrix_A = np.zeros(
            [dim * (len(self._knots) - 2), self._total_param_length]
        )
        smooth_matrix_B = np.zeros(dim * (len(self._knots) - 2))
        for i in range(len(self._knots) - 2):
            # Current connection knot
            # Since we are using relative_t for a knot, we need the relative_t to use the coefficient function as well
            relative_t = self._knots[i + 1] - self._knots[i]
            coeff = self.t_first_derivative_coefficient(relative_t)
            index_offset = i * dim * (self._spline_order + 1)
            next_index_offset = (i + 1) * dim * (self._spline_order + 1)
            for n in range(dim):
                for j in range(self._spline_order + 1):
                    smooth_matrix_A[i * dim + n][
                        index_offset + n * (self._spline_order + 1) + j
                    ] = coeff[j]
                smooth_matrix_A[i * dim + n][
                    next_index_offset + n * (self._spline_order + 1) + 1
                ] = -1
        return self._solver.add_equality_constraint(
            smooth_matrix_A, smooth_matrix_B
        )

    def _add_smooth_constraint_order_two(self) -> bool:
        assert (
            len(self._knots)
        ) >= 3, "_add_smooth_constraint_order_one error, knot size < 3"
        dim = self._dimension
        smooth_matrix_A = np.zeros(
            [dim * (len(self._knots) - 2), self._total_param_length]
        )
        smooth_matrix_B = np.zeros(dim * (len(self._knots) - 2))
        for i in range(len(self._knots) - 2):
            # Current connection knot
            # Since we are using relative_t for a knot, we need the relative_t to use the coefficient function as well
            relative_t = self._knots[i + 1] - self._knots[i]
            coeff = self.t_second_derivative_coefficient(relative_t)
            index_offset = i * dim * (self._spline_order + 1)
            next_index_offset = (i + 1) * dim * (self._spline_order + 1)
            for n in range(dim):
                for j in range(self._spline_order + 1):
                    smooth_matrix_A[i * dim + n][
                        index_offset + n * (self._spline_order + 1) + j
                    ] = coeff[j]
                smooth_matrix_A[i * dim + n][
                    next_index_offset + n * (self._spline_order + 1) + 2
                ] = -2
        return self._solver.add_equality_constraint(
            smooth_matrix_A, smooth_matrix_B
        )

    def _add_smooth_constraint_order_three(self) -> bool:
        assert (
            len(self._knots)
        ) >= 3, "_add_smooth_constraint_order_one error, knot size < 3"
        dim = self._dimension
        smooth_matrix_A = np.zeros(
            [dim * (len(self._knots) - 2), self._total_param_length]
        )
        smooth_matrix_B = np.zeros(dim * (len(self._knots) - 2))
        for i in range(len(self._knots) - 2):
            # Current connection knot
            # Since we are using relative_t for a knot, we need the relative_t to use the coefficient function as well
            relative_t = self._knots[i + 1] - self._knots[i]
            coeff = self.t_third_derivative_coefficient(relative_t)
            index_offset = i * dim * (self._spline_order + 1)
            next_index_offset = (i + 1) * dim * (self._spline_order + 1)
            for n in range(dim):
                for j in range(self._spline_order + 1):
                    smooth_matrix_A[i * dim + n][
                        index_offset + n * (self._spline_order + 1) + j
                    ] = coeff[j]
                smooth_matrix_A[i * dim + n][
                    next_index_offset + n * (self._spline_order + 1) + 3
                ] = -6
        return self._solver.add_equality_constraint(
            smooth_matrix_A, smooth_matrix_B
        )

    def apply_derivative_to_objective(
        self,
        coefficient_func,
        weight: float,
        t_ref: np.ndarray = None,
        point_ref: np.ndarray = None,
    ) -> bool:
        # If both t_ref and point_ref is not None, then they need to have same row size
        if hasattr(t_ref, "__len__") and hasattr(
            point_ref, "__len__"
        ):  # Need better way to tell if it's an np.ndarray
            point_ref = np.asarray(point_ref).reshape(-1, self._dimension)
            assert (
                t_ref.shape[0] == point_ref.shape[0]
            ), "Input ref-points and ref-t not matched!"

        # If point_ref is not None, then it needs to have second dimension == self._dimension (since this is nd point)
        if hasattr(point_ref, "__len__"):
            assert (
                point_ref.shape[1] == self._dimension
            ), "Input ref points dimension is not equal to self._dimension!"

        # If input t is None, then it means we only add at knots, we also need to check t_ref == point_ref in size
        if not hasattr(t_ref, "__len__") and not isinstance(t_ref, float):
            t_ref = self._knots
            assert point_ref.shape[0] == len(self._knots)

        # If input point_ref is None, then we construct a zero point_ref
        if not hasattr(point_ref, "__len__"):
            point_ref = np.zeros((len(t_ref), self._dimension))

        # J = 0.5 * X_tran * Hessian * X + X_tran * Gradient
        param_number = (
            self._spline_order + 1
        )  # param number for same knot same dimension
        for t_val, pt_val in zip(t_ref, point_ref):
            cur_index = self._search_prev_knot_index(t_val)
            cur_relative_t = t_val - self._knots[cur_index]
            cur_append_start_row = self._dimension * cur_index * param_number

            cur_gradient = []
            for n in range(self._dimension):
                cur_gradient.append(-2.0 * pt_val[n] * weight)

            for i in range(param_number):
                for n in range(self._dimension):
                    idx = cur_append_start_row + n * param_number
                    self._solver.add_value_to_gradient_vector(
                        i + idx, cur_gradient[n]
                    )
                    cur_gradient[n] *= cur_relative_t

            cur_hessian_matrix = np.zeros((param_number, param_number))
            cur_coefficient = coefficient_func(cur_relative_t)

            for i in range(param_number):
                for j in range(param_number):
                    cur_hessian_matrix[i][j] = (
                        2 * cur_coefficient[i] * cur_coefficient[j]
                    )

            for n in range(self._dimension):
                idx = cur_append_start_row + n * param_number
                self._solver.add_to_objective_function(
                    idx,
                    idx,
                    param_number,
                    param_number,
                    weight * cur_hessian_matrix,
                )

        return True

    def apply_upper_bound_with_order(derivative_function):
        def wrapper(self, t: float, *args, **kwargs):
            assert (
                kwargs == {}
            ), "apply_upper_bound_with_order error, do not use named input variable!"
            for arg in args:
                assert(isinstance(arg, (int, float))), "get argument not in float or int!"
            dim = self._dimension
            # Get index-offset and derivative coefficient from the function
            index, coeff = derivative_function(self, t, *args)

            assert (
                len([*args]) == dim
            ), "apply_upper_bound_with_order error, input boundary point size != dimension size"
            boundary_matrix_B = np.array([*args])
            boundary_matrix_A = np.zeros([dim, self._total_param_length])
            # Get the index-offset for the current boundary matrix
            index_offset = index * (self._spline_order + 1) * dim
            for n in range(dim):
                pos_offset = (self._spline_order + 1) * n
                for i in range(self._spline_order + 1):
                    boundary_matrix_A[n][
                        index_offset + pos_offset + i
                    ] = coeff[i]
            return self._solver.add_inequality_constraint(
                boundary_matrix_A, boundary_matrix_B
            )

        return wrapper

    def apply_lower_bound_with_order(derivative_function):
        def wrapper(self, t: float, *args, **kwargs):
            assert (
                kwargs == {}
            ), "apply_lower_bound_with_order error, do not use named input variable!"
            for arg in args:
                assert(isinstance(arg, (int, float))), "get argument not in float or int!"
            dim = self._dimension
            # Get index-offset and derivative coefficient from the function
            index, coeff = derivative_function(self, t, *args)

            assert (
                len([*args]) == dim
            ), "apply_lower_bound_with_order error, input boundary point size != dimension size"
            boundary_matrix_B = np.negative(
                np.array([*args])
            )  # negative sign for lower bound
            boundary_matrix_A = np.zeros([dim, self._total_param_length])
            # Get the index-offset for the current boundary matrix
            index_offset = index * (self._spline_order + 1) * dim
            for n in range(dim):
                pos_offset = (self._spline_order + 1) * n
                for i in range(self._spline_order + 1):
                    boundary_matrix_A[n][
                        index_offset + pos_offset + i
                    ] = -coeff[i]
            return self._solver.add_inequality_constraint(
                boundary_matrix_A, boundary_matrix_B
            )

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

    @apply_lower_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_lower_bound(self, t: float, *args, **kwargs):
        return self.t_coefficient(t)

    @apply_lower_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_first_derivative_lower_bound(
        self, t: float, *args, **kwargs
    ):
        return self.t_first_derivative_coefficient(t)

    @apply_lower_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_second_derivative_lower_bound(
        self, t: float, *args, **kwargs
    ):
        return self.t_second_derivative_coefficient(t)

    @apply_lower_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_third_derivative_lower_bound(
        self, t: float, *args, **kwargs
    ):
        return self.t_third_derivative_coefficient(t)

    @apply_upper_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_upper_bound(self, t: float, *args, **kwargs):
        return self.t_coefficient(t)

    @apply_upper_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_first_derivative_upper_bound(
        self, t: float, *args, **kwargs
    ):
        return self.t_first_derivative_coefficient(t)

    @apply_upper_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_second_derivative_upper_bound(
        self, t: float, *args, **kwargs
    ):
        return self.t_second_derivative_coefficient(t)

    @apply_upper_bound_with_order
    @get_knot_index_and_coefficient
    def add_point_third_derivative_upper_bound(
        self, t: float, *args, **kwargs
    ):
        return self.t_third_derivative_coefficient(t)

    def add_reference_points_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray
    ) -> bool:
        return self.apply_derivative_to_objective(
            self.t_coefficient, weight, t_ref, points_ref
        )

    def add_first_derivative_points_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray = None
    ) -> bool:
        return self.apply_derivative_to_objective(
            self.t_first_derivative_coefficient, weight, t_ref, points_ref
        )

    def add_second_derivative_points_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray = None
    ) -> bool:
        return self.apply_derivative_to_objective(
            self.t_second_derivative_coefficient, weight, t_ref, points_ref
        )

    def add_third_derivative_points_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray = None
    ) -> bool:
        return self.apply_derivative_to_objective(
            self.t_third_derivative_coefficient, weight, t_ref, points_ref
        )
