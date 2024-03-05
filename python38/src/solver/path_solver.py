import numpy as np
import math
from typing import List
from src.abstract import Abstract
from src.solver.spline_nd_solver import SplineNdSolver


class PathSolver(SplineNdSolver, Abstract):
    def __init__(self, knots: List[float], spline_order: int):
        super(PathSolver, self).__init__(
            knots=knots, spline_order=spline_order, dimension=2
        )

    @classmethod
    def from_knots_and_order(cls, knots: List[float], spline_order: int):
        return cls(knots, spline_order)

    def add_angle_constraint(self, t: float, angle: float) -> bool:
        index = self._search_prev_knot_index(t=t)
        relative_time = t - self._knots[index]
        coeff = self.t_first_derivative_coefficient(relative_time)

        sin = math.sin(angle)
        cos = math.cos(angle)
        matrix_A = np.zeros([1, self._total_param_length])
        matrix_B = np.zeros(1)

        coeff_offset_index = self._dimension * (self._spline_order + 1) * index
        for i in range(self._spline_order + 1):
            matrix_A[0][coeff_offset_index + i] = -sin * coeff[i]
            matrix_A[0][coeff_offset_index + self._spline_order + 1 + i] = (
                cos * coeff[i]
            )

        if not self._solver.add_equality_constraint(matrix_A, matrix_B):
            return False

        # Add inequality constraint to determine the sign
        while angle < 0:
            angle += 2 * math.pi

        while angle >= 2 * math.pi:
            angle -= 2 * math.pi

        inequality_matrix_A = np.zeros([2, self._total_param_length])
        inequality_matrix_B = np.zeros(2)

        x_sign = 1
        y_sign = 1
        if angle > math.pi / 2 and angle < math.pi * 1.5:
            x_sign = -1
        if angle > math.pi:
            y_sign = -1

        for i in range(self._spline_order + 1):
            inequality_matrix_A[0][coeff_offset_index + i] = coeff[i] * -x_sign
            inequality_matrix_A[1][
                coeff_offset_index + self._spline_order + 1 + i
            ] = (coeff[i] * -y_sign)
        return self._solver.add_inequality_constraint(
            inequality_matrix_A, inequality_matrix_B
        )

    def add_angle_kernel(
        self, t: float, angle_ref: float, weight: float
    ) -> bool:
        index = self._search_prev_knot_index(t=t)
        relative_time = t - self._knots[index]
        coeff = self.t_first_derivative_coefficient(relative_time)

        sin = math.sin(angle)
        cos = math.cos(angle)

        total_param_number = self._dimension * (
            self._spline_order + 1
        )  # dimension = 2
        cur_hessian_matrix = np.zeros((total_param_number, total_param_number))

        param_number = self._spline_order + 1
        expanded_coeff_matrix = np.zeros((param_number, param_number))
        for i in range(param_number):
            expanded_coeff_matrix[i][i] = coeff[i] * coeff[i]
            for j in range(i + 1, param_number):
                expanded_coeff_matrix[i][j] = coeff[i] * coeff[j]
                expanded_coeff_matrix[j][i] = expanded_coeff_matrix[i][j]

        cur_hessian_matrix[0 : 0 + param_number, 0 : 0 + param_number] += (
            expanded_coeff_matrix * sin * sin
        )
        cur_hessian_matrix[
            param_number : 2 * param_number, param_number : 2 * param_number
        ] += (expanded_coeff_matrix * cos * cos)
        cur_hessian_matrix[
            0 : 0 + param_number, param_number : 2 * param_number
        ] += (expanded_coeff_matrix * sin * cos)
        cur_hessian_matrix[
            param_number : 2 * param_number, 0 : 0 + param_number
        ] += (expanded_coeff_matrix * sin * cos)

        objective_hessian_start_index = (
            self._dimension * (self._spline_order + 1) * index
        )  # dimension = 2
        return self._solver.add_to_objective_function(
            objective_hessian_start_index,
            objective_hessian_start_index,
            total_param_number,
            total_param_number,
            weight * cur_hessian_matrix,
        )

    def add_2d_boundary_constraint(
        self,
        t: float,
        ref_x: float,
        ref_y: float,
        ref_heading: float,
        longitudinal_bound: float,
        lateral_bound: float,
    ) -> bool:
        index = self._search_prev_knot_index(t=t)
        relative_time = t - self._knots[index]
        coeff = self.t_coefficient(relative_time)

        sin = math.sin(ref_heading)
        cos = math.cos(ref_heading)

        inequality_matrix_A = np.zeros([4, self._total_param_length])
        inequality_matrix_B = np.array(
            [
                longitudinal_bound + cos * ref_x + sin * ref_y,
                longitudinal_bound - cos * ref_x - sin * ref_y,
                lateral_bound - sin * ref_x + cos * ref_y,
                lateral_bound + sin * ref_x - cos * ref_y,
            ]
        )

        coeff_offset_index = self._dimension * (self._spline_order + 1) * index
        for i in range(self._spline_order + 1):
            inequality_matrix_A[0][coeff_offset_index + i] = cos * coeff[i]
            inequality_matrix_A[0][
                coeff_offset_index + self._spline_order + 1 + i
            ] = (sin * coeff[i])

            inequality_matrix_A[1][coeff_offset_index + i] = -cos * coeff[i]
            inequality_matrix_A[1][
                coeff_offset_index + self._spline_order + 1 + i
            ] = (-sin * coeff[i])

            inequality_matrix_A[2][coeff_offset_index + i] = -sin * coeff[i]
            inequality_matrix_A[2][
                coeff_offset_index + self._spline_order + 1 + i
            ] = (cos * coeff[i])

            inequality_matrix_A[3][coeff_offset_index + i] = sin * coeff[i]
            inequality_matrix_A[3][
                coeff_offset_index + self._spline_order + 1 + i
            ] = (-cos * coeff[i])
        # print("Matrix A:", inequality_matrix_A[0][0], ":", cos * coeff[0])
        # print("Matrix B:", inequality_matrix_B[0])

        return self._solver.add_inequality_constraint(
            inequality_matrix_A, inequality_matrix_B
        )
