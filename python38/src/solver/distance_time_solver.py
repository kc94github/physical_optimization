import numpy as np
from typing import List
from src.abstract import Abstract
from src.solver.spline_nd_solver import SplineNdSolver


class DistanceTimeSolver(SplineNdSolver, Abstract):
    def __init__(self, knots: List[float], spline_order: int):
        super(DistanceTimeSolver, self).__init__(
            knots=knots, spline_order=spline_order, dimension=1
        )

    @classmethod
    def from_knots_and_order(cls, knots: List[float], spline_order: int):
        return cls(knots, spline_order)

    def add_distance_point_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray
    ):
        return self.add_reference_points_to_objective(
            weight, t_ref, points_ref
        )

    def add_speed_point_penalty_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray = None
    ):
        return self.add_first_derivative_points_to_objective(
            weight, t_ref, points_ref
        )

    def add_acceleration_point_penalty_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray = None
    ):
        return self.add_second_derivative_points_to_objective(
            weight, t_ref, points_ref
        )

    def add_jerk_point_penalty_to_objective(
        self, weight: float, t_ref: np.ndarray, points_ref: np.ndarray = None
    ):
        return self.add_third_derivative_points_to_objective(
            weight, t_ref, points_ref
        )

    def add_distance_increasing_monotone(
        self, t_ref: np.ndarray = None
    ) -> bool:
        if not hasattr(t_ref, "__len__") and not isinstance(t_ref, float):
            t_ref = self._knots

        if isinstance(t_ref, float):
            t_ref = [t_ref]

        for t in t_ref:
            if not self.add_speed_lower_bound(t, 0):
                return False
        return True

    def add_distance_constraint(self, t: float, distance: float) -> bool:
        if hasattr(distance, "__len__"):
            for t_val, d_val in zip(t, distance):
                if self.add_point_constraint(t_val, d_val):
                    return False
            return True
        return self.add_point_constraint(t, distance)

    def add_speed_constraint(self, t: float, speed: float) -> bool:
        if hasattr(speed, "__len__"):
            for t_val, s_val in zip(t, speed):
                if self.add_point_first_derivative_constraint(t_val, s_val):
                    return False
            return True
        return self.add_point_first_derivative_constraint(t, speed)

    def add_acceleration_constraint(
        self, t: float, acceleration: float
    ) -> bool:
        if hasattr(acceleration, "__len__"):
            for t_val, a_val in zip(t, acceleration):
                if self.add_point_second_derivative_constraint(t_val, a_val):
                    return False
            return True
        return self.add_point_second_derivative_constraint(t, acceleration)

    def add_jerk_constraint(self, t: float, jerk: float) -> bool:
        if hasattr(jerk, "__len__"):
            for t_val, j_val in zip(t, jerk):
                if self.add_point_third_derivative_constraint(t_val, j_val):
                    return False
            return True
        return self.add_point_third_derivative_constraint(t, jerk)

    def add_distance_lower_bound(self, t: float, distance: float) -> bool:
        if hasattr(distance, "__len__"):
            for t_val, d_val in zip(t, distance):
                if self.add_point_lower_bound(t_val, d_val):
                    return False
            return True
        return self.add_point_lower_bound(t, distance)

    def add_speed_lower_bound(self, t: float, speed: float) -> bool:
        if hasattr(speed, "__len__"):
            for t_val, s_val in zip(t, speed):
                if self.add_point_first_derivative_lower_bound(t_val, s_val):
                    return False
            return True
        return self.add_point_first_derivative_lower_bound(t, speed)

    def add_acceleration_lower_bound(
        self, t: float, acceleration: float
    ) -> bool:
        if hasattr(acceleration, "__len__"):
            for t_val, a_val in zip(t, acceleration):
                if self.add_point_second_derivative_lower_bound(t_val, a_val):
                    return False
            return True
        return self.add_point_second_derivative_lower_bound(t, acceleration)

    def add_jerk_lower_bound(self, t: float, jerk: float) -> bool:
        if hasattr(jerk, "__len__"):
            for t_val, j_val in zip(t, jerk):
                if self.add_point_third_derivative_lower_bound(t_val, j_val):
                    return False
            return True
        return self.add_point_third_derivative_lower_bound(t, jerk)

    def add_distance_upper_bound(self, t: float, distance: float) -> bool:
        if hasattr(distance, "__len__"):
            for t_val, d_val in zip(t, distance):
                if self.add_point_upper_bound(t_val, d_val):
                    return False
            return True
        return self.add_point_upper_bound(t, distance)

    def add_speed_upper_bound(self, t: float, speed: float) -> bool:
        if hasattr(speed, "__len__"):
            for t_val, s_val in zip(t, speed):
                if self.add_point_first_derivative_upper_bound(t_val, s_val):
                    return False
            return True
        return self.add_point_first_derivative_upper_bound(t, speed)

    def add_acceleration_upper_bound(
        self, t: float, acceleration: float
    ) -> bool:
        if hasattr(acceleration, "__len__"):
            for t_val, a_val in zip(t, acceleration):
                if self.add_point_second_derivative_upper_bound(t_val, a_val):
                    return False
            return True
        return self.add_point_second_derivative_upper_bound(t, acceleration)

    def add_jerk_upper_bound(self, t: float, jerk: float) -> bool:
        if hasattr(jerk, "__len__"):
            for t_val, j_val in zip(t, jerk):
                if self.add_point_third_derivative_upper_bound(t_val, j_val):
                    return False
            return True
        return self.add_point_third_derivative_upper_bound(t, jerk)
