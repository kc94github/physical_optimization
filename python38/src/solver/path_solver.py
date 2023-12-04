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

        return self._solver.add_inequality_constraint(
            inequality_matrix_A, inequality_matrix_B
        )
