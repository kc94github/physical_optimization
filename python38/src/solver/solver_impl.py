import numpy as np
from typing import List, Tuple
from qpsolvers import solve_qp, Problem, solve_problem
from qpsolvers.solution import Solution
from src.abstract import Abstract


class SolverImpl(Abstract):
    def __init__(self, size: int = 1):
        self._size = size
        self.init_matrices()

    def init_matrices(self):
        self._hessian_matrix = np.zeros([self._size, self._size])
        self._gradient_vector = np.zeros((self._size))
        self._equality_constraint_matrix = None
        self._inequality_constraint_matrix = None
        self._equality_constraint_vector = None
        self._inequality_constraint_vector = None
        self._lower_bound_vector = None
        self._upper_bound_vector = None

    def __repr__(self):
        return f"QP Problem: \n Size: {self._size} \n Hessian: {self._hessian_matrix},{self._gradient_vector} \n Eq_constraint: {self._equality_constraint_matrix},{self._equality_constraint_vector} \n Ineq_constraint: {self._inequality_constraint_matrix},{self._inequality_constraint_vector} \n Lower/Upper bound: {self._lower_bound_vector},{self._upper_bound_vector}"

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self.size()

    @property
    def hessian_matrix(self) -> np.ndarray:
        return self._hessian_matrix

    @property
    def gradient_vector(self) -> np.ndarray:
        return self._gradient_vector

    @property
    def equality_constraint_matrix(self) -> np.ndarray:
        return self._equality_constraint_matrix

    @property
    def inequality_constraint_matrix(self) -> np.ndarray:
        return self._inequality_constraint_matrix

    @property
    def equality_constraint_vector(self) -> np.ndarray:
        return self._equality_constraint_vector

    @property
    def inequality_constraint_vector(self) -> np.ndarray:
        return self._inequality_constraint_vector

    @property
    def lower_bound_vector(self) -> np.ndarray:
        return self._lower_bound_vector

    @property
    def upper_bound_vector(self) -> np.ndarray:
        return self._upper_bound_vector

    def add_to_objective_function(
        self,
        start_row: int,
        start_col: int,
        add_row_size: int,
        add_col_size: int,
        add_hessian_submatrix: np.ndarray,
        add_gradient_subvector: np.ndarray = None,
    ):
        assert add_row_size == add_hessian_submatrix.shape[0]
        assert add_col_size == add_hessian_submatrix.shape[1]
        self._hessian_matrix[
            start_row : start_row + add_row_size,
            start_col : start_col + add_col_size,
        ] += add_hessian_submatrix

        if add_gradient_subvector != None and add_gradient_subvector.ndim == 2:
            add_gradient_subvector = add_gradient_subvector.reshape((-1,))
            assert add_row_size == add_gradient_subvector.shape[0]
            self._gradient_vector[
                start_row : start_row + add_row_size
            ] += add_gradient_subvector
        return True

    def set_objective_function(
        self, hessian_matrix: np.ndarray, gradient_vector: np.ndarray
    ) -> bool:
        if (
            hessian_matrix.shape[0] != hessian_matrix.shape[1]
            or hessian_matrix.shape[0] != self._hessian_matrix.shape[0]
        ):
            raise Exception("input hessian matrix size not matching set size!")

        if gradient_vector.ndim == 2:
            gradient_vector = gradient_vector.reshape((-1,))

        if hessian_matrix.shape[0] != gradient_vector.shape[0]:
            raise Exception(
                "input hessian matrix size not matching gradient vector size!"
            )

        self._hessian_matrix = hessian_matrix
        self._gradient_vector = gradient_vector
        return True

    def set_upper_bound(self, upper_bound: np.ndarray) -> bool:
        if upper_bound.ndim == 2:
            upper_bound = upper_bound.reshape((-1,))
        if upper_bound.shape[0] != self._size:
            raise Exception(
                "input upper bound vector size not matching set size!"
            )
        self._upper_bound_vector = upper_bound
        return True

    def set_lower_bound(self, lower_bound: np.ndarray) -> bool:
        if lower_bound.ndim == 2:
            lower_bound = lower_bound.reshape((-1,))
        if lower_bound.shape[0] != self._size:
            raise Exception(
                "input lower bound vector size not matching set size!"
            )
        self._lower_bound_vector = lower_bound
        return True

    def add_equality_constraint(
        self, equality_matrix: np.ndarray, equality_vector: np.ndarray
    ) -> bool:
        (
            self._equality_constraint_matrix,
            self._equality_constraint_vector,
        ) = SolverImpl._by_adding_constraint_condition(
            equality_matrix,
            equality_vector,
            self._equality_constraint_matrix,
            self._equality_constraint_vector,
            self._size,
        )
        return True

    def add_inequality_constraint(
        self, inequality_matrix: np.ndarray, inequality_vector: np.ndarray
    ) -> bool:
        (
            self._inequality_constraint_matrix,
            self._inequality_constraint_vector,
        ) = SolverImpl._by_adding_constraint_condition(
            inequality_matrix,
            inequality_vector,
            self._inequality_constraint_matrix,
            self._inequality_constraint_vector,
            self._size,
        )
        return True

    @staticmethod
    def _by_adding_constraint_condition(
        coefficient_matrix: np.ndarray,
        constant_vector: np.ndarray,
        total_constraint_matrix: np.ndarray,
        total_constraint_vector: np.ndarray,
        size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if coefficient_matrix.ndim == 1:
            # typically it only happens if it only got one condition to add, and it will be set row-wise
            # it needs to be reshape to be col-wise
            coefficient_matrix = coefficient_matrix.reshape((1, -1))

        if constant_vector.ndim == 2:
            constant_vector = constant_vector.reshape((-1,))

        if coefficient_matrix.shape[0] != constant_vector.shape[0]:
            raise Exception(
                "input coefficient matrix and constant vector row size not matching!"
            )

        if coefficient_matrix.shape[1] != size:
            raise Exception(
                "input coefficient matrix col size not matching problem size!"
            )

        if not isinstance(total_constraint_matrix, np.ndarray):
            return coefficient_matrix, constant_vector

        if total_constraint_matrix.shape[1] != coefficient_matrix.shape[1]:
            raise Exception(
                "input coefficient matrix not compatible with existing constraint matrix!"
            )

        return np.concatenate(
            (total_constraint_matrix, coefficient_matrix), axis=0
        ), np.concatenate((total_constraint_vector, constant_vector), axis=0)

    def solve(
        self, solver_name: str = "osqp", solution_only: bool = True
    ) -> Solution:
        if np.equal(
            self._hessian_matrix, np.zeros([self._size, self._size])
        ).all():
            self._hessian_matrix = 0.01 * np.eye(self._size)

        if not isinstance(self._gradient_vector, np.ndarray):
            self._gradient_vector = np.zeros(self._size, 1)

        problem = Problem(
            self._hessian_matrix,
            self._gradient_vector,
            self._inequality_constraint_matrix,
            self._inequality_constraint_vector,
            self._equality_constraint_matrix,
            self._equality_constraint_vector,
            self._lower_bound_vector,
            self._upper_bound_vector,
        )
        solution = solve_problem(problem, solver=solver_name)
        if solution_only:
            return solution.x
        return solution
