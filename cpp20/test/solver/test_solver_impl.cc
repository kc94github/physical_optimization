#include "solver_impl.h"
#include <gtest/gtest.h>

TEST(SolverImplTest, BasicTest) {
  Solver::SolverImpl impl(4);
  EXPECT_EQ(impl.size(), 4);

  EXPECT_EQ(impl.hessian_matrix().rows(), 4);
  EXPECT_EQ(impl.hessian_matrix().cols(), 4);

  EXPECT_EQ(impl.gradient_vector().rows(), 4);

  EXPECT_EQ(impl.constraint_matrix().rows(), 0);
  EXPECT_EQ(impl.constraint_matrix().cols(), 4);

  EXPECT_EQ(impl.upper_bound_vector().rows(), 0);
  EXPECT_EQ(impl.lower_bound_vector().rows(), 0);
}

TEST(SolverImplTest, SetMatrixTest) {
  Eigen::MatrixXd hessian_matrix(8, 8);
  hessian_matrix << 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.5301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.5606, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd gradient_vector(8);
  gradient_vector << -20.0, 0.0, -18.5823, 0.0, -17.1932, 0.0, 0.0, 0.0;

  Eigen::MatrixXd equality_matrix(6, 8);
  equality_matrix << -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0355329, 1.0, -1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0355329, 0.0, -1.0, 0.0, 1.0;

  Eigen::VectorXd equality_vector(6);
  equality_vector.setZero();

  Solver::SolverImpl impl(8);
  EXPECT_EQ(impl.size(), 8);

  impl.set_objective_function(hessian_matrix, gradient_vector);

  Eigen::SparseMatrix<double> converted_hessian_matrix =
      hessian_matrix.sparseView();
  EXPECT_TRUE(converted_hessian_matrix.isApprox(impl.hessian_matrix()));
  EXPECT_EQ(impl.gradient_vector(), gradient_vector);

  impl.add_equality_constraint(equality_matrix, equality_vector);

  Eigen::SparseMatrix<double> converted_equality_matrix =
      equality_matrix.sparseView();
  EXPECT_TRUE(converted_equality_matrix.isApprox(impl.constraint_matrix()));
  EXPECT_EQ(impl.upper_bound_vector(), equality_vector);
  EXPECT_EQ(impl.lower_bound_vector(), equality_vector);

  impl.reset_matrices();
  impl.add_to_objective_function_from_sparse(
      0, 0, 8, 8, converted_hessian_matrix, gradient_vector);
  EXPECT_TRUE(converted_hessian_matrix.isApprox(impl.hessian_matrix()));
  EXPECT_EQ(impl.gradient_vector(), gradient_vector);
}

TEST(SolverImplTest, MatrixSolverExampleTest) {
  Eigen::MatrixXd hessian_matrix(8, 8);
  hessian_matrix << 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.5301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.5606, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd gradient_vector(8);
  gradient_vector << -20.0, 0.0, -18.5823, 0.0, -17.1932, 0.0, 0.0, 0.0;

  Eigen::MatrixXd equality_matrix(6, 8);
  equality_matrix << -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0355329, 1.0, -1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0355329, 0.0, -1.0, 0.0, 1.0;

  Eigen::VectorXd equality_vector(6);
  equality_vector.setZero();

  Solver::SolverImpl impl(8);
  EXPECT_EQ(impl.size(), 8);

  impl.set_objective_function(hessian_matrix, gradient_vector);

  Eigen::SparseMatrix<double> converted_hessian_matrix =
      hessian_matrix.sparseView();
  EXPECT_TRUE(converted_hessian_matrix.isApprox(impl.hessian_matrix()));
  EXPECT_EQ(impl.gradient_vector(), gradient_vector);

  impl.add_equality_constraint(equality_matrix, equality_vector);

  Eigen::SparseMatrix<double> converted_equality_matrix =
      equality_matrix.sparseView();
  EXPECT_TRUE(converted_equality_matrix.isApprox(impl.constraint_matrix()));
  EXPECT_EQ(impl.upper_bound_vector(), equality_vector);
  EXPECT_EQ(impl.lower_bound_vector(), equality_vector);

  Eigen::VectorXd result = impl.solve();
  EXPECT_EQ(result.rows(), 8);

  Eigen::VectorXd ground_truth_result(8);
  ground_truth_result << 3.67653e-07, 7.58958e-07, 1.51793e-06, 0.562595,
      0.562595, 0.0199906, 0.562595, -3.57043e-11;
  EXPECT_TRUE(ground_truth_result.isApprox(result, 1e-3));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}