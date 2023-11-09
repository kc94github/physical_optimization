import unittest
import numpy as np
from qpsolvers import solve_qp, Problem, solve_problem
from qpsolvers.solution import Solution

from src.solver.solver_base import SolverBase

class TestSolverBase(unittest.TestCase):
    def test_basic(self):
        s = SolverBase(10)
        print(s)
        self.assertEqual(s.size(), 10)
        self.assertEqual(len(s), 10)

    def test_interpret(self):
        M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
        P = M.T.dot(M)  # quick way to build a symmetric matrix
        q = np.array([3., 2., 3.]).dot(M).reshape((3,1))
        G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
        h = np.array([3., 2., -2.]).reshape((3,1))
        A = np.array([1., 1., 1.])
        b = np.array([1.])
        lb = -0.6 * np.ones(3)
        ub = +0.7 * np.ones(3)

        problem = Problem(P, q, G, h, A, b, lb, ub)
        solution = solve_problem(problem, solver="osqp")
        x_result = solution.x

        self.assertEqual(solution.found, True)
        self.assertAlmostEqual(x_result[0], 0.63320, 3)
        self.assertAlmostEqual(x_result[1], -0.3333, 3)
        self.assertAlmostEqual(x_result[2], 0.7001, 3)

        s = SolverBase(3)
        s.set_objective_function(P, q)
        s.add_equality_constraint(A, b)
        s.add_inequality_constraint(G, h)
        s.set_lower_bound(lb)
        s.set_upper_bound(ub)
        solution_2 = s.solve(solver_name = "osqp", solution_only = False)
        x_result_2 = solution_2.x
        self.assertAlmostEqual(x_result_2[0], x_result[0], 3)
        self.assertAlmostEqual(x_result_2[1], x_result[1], 3)
        self.assertAlmostEqual(x_result_2[2], x_result[2], 3)

