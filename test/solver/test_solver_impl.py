import unittest
import numpy as np
from qpsolvers import solve_qp, Problem, solve_problem
from qpsolvers.solution import Solution

from src.solver.solver_impl import SolverImpl


class TestSolverImpl(unittest.TestCase):
    def test_basic(self):
        s = SolverImpl(10)
        print(s)
        self.assertEqual(s.size(), 10)
        self.assertEqual(len(s), 10)

    def test_interpret(self):
        # Example from official documentation
        M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        P = M.T.dot(M)  # quick way to build a symmetric matrix
        q = np.array([3.0, 2.0, 3.0]).dot(M).reshape((3, 1))
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3, 1))
        A = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0])
        lb = -0.6 * np.ones(3)
        ub = +0.7 * np.ones(3)

        problem = Problem(P, q, G, h, A, b, lb, ub)
        solution = solve_problem(problem, solver="osqp")
        x_result = solution.x

        self.assertEqual(solution.found, True)
        self.assertAlmostEqual(x_result[0], 0.63320, 2)
        self.assertAlmostEqual(x_result[1], -0.3333, 2)
        self.assertAlmostEqual(x_result[2], 0.7001, 2)

        s = SolverImpl(3)
        s.set_objective_function(P, q)
        s.add_equality_constraint(A, b)
        s.add_inequality_constraint(G, h)
        s.set_lower_bound(lb)
        s.set_upper_bound(ub)
        solution_2 = s.solve(solver_name="osqp", solution_only=False)
        x_result_2 = solution_2.x
        self.assertAlmostEqual(x_result_2[0], x_result[0], 2)
        self.assertAlmostEqual(x_result_2[1], x_result[1], 2)
        self.assertAlmostEqual(x_result_2[2], x_result[2], 2)

    def test_objective_func(self):
        # Example from official documentation
        M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        P = M.T.dot(M)  # quick way to build a symmetric matrix
        q = np.array([3.0, 2.0, 3.0]).dot(M).reshape((3, 1))
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3, 1))
        A = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0])
        lb = -0.6 * np.ones(3)
        ub = +0.7 * np.ones(3)

        s = SolverImpl(3)
        s.set_objective_function(P, q)
        self.assertEqual(s.hessian_matrix.all(), P.all())

        added = np.array([[3.0], [5.0]])  # 2*1, this is OK
        s.add_to_objective_function(
            start_row=1,
            start_col=2,
            add_row_size=2,
            add_col_size=1,
            add_hessian_submatrix=added,
        )
        P[1:3, 2:3] += added
        self.assertEqual(s.hessian_matrix.all(), P.all())

        with self.assertRaises(AssertionError):  # Create size mismatched error
            s.add_to_objective_function(
                start_row=1,
                start_col=2,
                add_row_size=2,
                add_col_size=2,
                add_hessian_submatrix=added,
            )

        with self.assertRaises(ValueError):
            added2 = np.array(
                [[3.0, 7.0], [5.0, 2.0]]
            )  # 2*2, this will create error
            s.add_to_objective_function(
                start_row=1,
                start_col=2,
                add_row_size=2,
                add_col_size=2,
                add_hessian_submatrix=added2,
            )
