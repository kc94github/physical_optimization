#include "path_solver.h"
#include <gtest/gtest.h>

class PathSolverTest : public ::testing::Test {
public:
  static std::vector<double> knots() { return {0.0, 1.0, 2.0}; }

  static std::vector<double> test_t() {
    return {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0};
  }

  static std::vector<double> test_x() {
    return {1.0, 1.75, 3.25, 5.875, 10.0, 16.0, 24.25, 35.125, 49.0};
  }

  static std::vector<double> test_x_first_derivative() {
    return {2.0, 4.25, 8.0, 13.25, 20.0, 28.25, 38.0, 49.25, 62.0};
  }

  static std::vector<double> test_x_second_derivative() {
    return {6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0};
  }

  static std::vector<double> test_x_third_derivative() {
    return {24, 24, 24, 24, 24, 24, 24, 24, 24};
  }

  static std::vector<double> test_y() {
    return {5.0,      5.265625, 5.625,     6.171875, 7.0,
            8.203125, 9.875,    12.109375, 15.0};
  }

  static std::vector<std::vector<double>> test_pts() {
    std::vector<std::vector<double>> rst;
    for (int i = 0; i < PathSolverTest::test_y().size(); i++) {
      rst.push_back({PathSolverTest::test_x()[i], PathSolverTest::test_y()[i]});
    }
    return rst;
  }

  static std::vector<std::vector<double>> test_pts_zeros() {
    std::vector<std::vector<double>> rst;
    for (int i = 0; i < PathSolverTest::test_y().size(); i++) {
      rst.push_back({0.0, 0.0});
    }
    return rst;
  }

  static std::vector<double> test_y_first_derivative() {
    return {1.0, 1.1875, 1.75, 2.6875, 4.0, 5.6875, 7.75, 10.1875, 13.0};
  }

  static std::vector<double> test_y_second_derivative() {
    return {0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0};
  }

  static std::vector<double> test_y_third_derivative() {
    return {6, 6, 6, 6, 6, 6, 6, 6, 6};
  }

  static void assertSolverResult(Eigen::VectorXd rst, double precision) {
    EXPECT_NEAR(rst[0], 1.0, precision);
    EXPECT_NEAR(rst[1], 2.0, precision);
    EXPECT_NEAR(rst[2], 3.0, precision);
    EXPECT_NEAR(rst[3], 4.0, precision);
    EXPECT_NEAR(rst[4], 5.0, precision);
    EXPECT_NEAR(rst[5], 1.0, precision);
    EXPECT_NEAR(rst[6], 0.0, precision);
    EXPECT_NEAR(rst[7], 1.0, precision);
    EXPECT_NEAR(rst[8], 10.0, precision);
    EXPECT_NEAR(rst[9], 20.0, precision);
    EXPECT_NEAR(rst[10], 15.0, precision);
    EXPECT_NEAR(rst[11], 4.0, precision);
    EXPECT_NEAR(rst[12], 7.0, precision);
    EXPECT_NEAR(rst[13], 4.0, precision);
    EXPECT_NEAR(rst[14], 3.0, precision);
    EXPECT_NEAR(rst[15], 1.0, precision);
  }
};

TEST_F(PathSolverTest, BasicTest) {
  Solver::PathSolver solver(this->knots(), 3);
  EXPECT_EQ(solver.size(), 16);

  EXPECT_EQ(solver.hessian_matrix().rows(), 16);
  EXPECT_EQ(solver.hessian_matrix().cols(), 16);

  EXPECT_EQ(solver.gradient_vector().rows(), 16);

  EXPECT_EQ(solver.dimension(), 2);
  EXPECT_EQ(solver.param_size(), 16);
}

TEST_F(PathSolverTest, BoundaryConstarintTest) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]

  // Y(t) = t^3 + t + 5 for [0-1]
  // Y(t) = (t-1)^3 + 3(t-1)^2 + 4(t-1) + 7 for [1-2]
  double longitudinal_bound = 0.01, lateral_bound = 0.01;
  Solver::PathSolver solver(this->knots(), 3);
  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_point_constraint(this->test_t().at(i),
                                {this->test_x().at(i), this->test_y().at(i)});
    double heading = std::atan2(this->test_y_first_derivative().at(i),
                                this->test_x_first_derivative().at(i));
    solver.add_2d_boundary_constraint(
        this->test_t().at(i), this->test_x().at(i), this->test_y().at(i),
        heading, longitudinal_bound, lateral_bound);
  }

  Eigen::VectorXd rst = solver.solve();
  this->assertSolverResult(rst, 0.001);
}

TEST_F(PathSolverTest, HeadingConstraintTest) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]

  // Y(t) = t^3 + t + 5 for [0-1]
  // Y(t) = (t-1)^3 + 3(t-1)^2 + 4(t-1) + 7 for [1-2]
  Solver::PathSolver solver(this->knots(), 3);
  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_point_constraint(this->test_t().at(i),
                                {this->test_x().at(i), this->test_y().at(i)});
    double heading = std::atan2(this->test_y_first_derivative().at(i),
                                this->test_x_first_derivative().at(i));
    solver.add_angle_constraint(this->test_t().at(i), heading);
  }

  Eigen::VectorXd rst = solver.solve();
  this->assertSolverResult(rst, 0.001);
}