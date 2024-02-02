#include "distance_time_solver.h"
#include <gtest/gtest.h>

class DistanceTimeSolverTest : public ::testing::Test {
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

  static std::vector<std::vector<double>> test_pts() {
    std::vector<std::vector<double>> rst;
    for (int i = 0; i < DistanceTimeSolverTest::test_x().size(); i++) {
      rst.push_back({DistanceTimeSolverTest::test_x()[i]});
    }
    return rst;
  }

  static void assertSolverResult(Eigen::VectorXd rst, double precision) {
    EXPECT_NEAR(rst[0], 1.0, precision);
    EXPECT_NEAR(rst[1], 2.0, precision);
    EXPECT_NEAR(rst[2], 3.0, precision);
    EXPECT_NEAR(rst[3], 4.0, precision);
    EXPECT_NEAR(rst[4], 10.0, precision);
    EXPECT_NEAR(rst[5], 20.0, precision);
    EXPECT_NEAR(rst[6], 15.0, precision);
    EXPECT_NEAR(rst[7], 4.0, precision);
  }
};

TEST_F(DistanceTimeSolverTest, BasicTest) {
  Solver::DistanceTimeSolver solver(this->knots(), 3);
  EXPECT_EQ(solver.size(), 8);

  EXPECT_EQ(solver.hessian_matrix().rows(), 8);
  EXPECT_EQ(solver.hessian_matrix().cols(), 8);

  EXPECT_EQ(solver.gradient_vector().rows(), 8);

  EXPECT_EQ(solver.dimension(), 1);
  EXPECT_EQ(solver.param_size(), 8);
}

TEST_F(DistanceTimeSolverTest, EvalTest) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
  Solver::DistanceTimeSolver solver(this->knots(), 3);
  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_distance_constraint(this->test_t().at(i), this->test_x().at(i));
  }

  Eigen::VectorXd rst = solver.solve();
  this->assertSolverResult(rst, 0.001);
}

TEST_F(DistanceTimeSolverTest, EvalDerivativesTest) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
  Solver::DistanceTimeSolver solver(this->knots(), 3);
  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_distance_constraint(this->test_t().at(i), this->test_x().at(i));
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_speed_constraint(this->test_t().at(i),
                                this->test_x_first_derivative().at(i));
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_acceleration_constraint(this->test_t().at(i),
                                       this->test_x_second_derivative().at(i));
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_jerk_constraint(this->test_t().at(i),
                               this->test_x_third_derivative().at(i));
  }

  Eigen::VectorXd rst = solver.solve();
  this->assertSolverResult(rst, 0.001);
}

TEST_F(DistanceTimeSolverTest, EvalDerivativesAndBoundsTest) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
  Solver::DistanceTimeSolver solver(this->knots(), 3);
  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_distance_constraint(this->test_t().at(i), this->test_x().at(i));
    solver.add_distance_lower_bound(this->test_t().at(i),
                                    this->test_x().at(i) - 0.01);
    solver.add_distance_upper_bound(this->test_t().at(i),
                                    this->test_x().at(i) + 0.01);
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_speed_constraint(this->test_t().at(i),
                                this->test_x_first_derivative().at(i));
    solver.add_speed_lower_bound(this->test_t().at(i),
                                 this->test_x_first_derivative().at(i) - 0.01);
    solver.add_speed_upper_bound(this->test_t().at(i),
                                 this->test_x_first_derivative().at(i) + 0.01);
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_acceleration_constraint(this->test_t().at(i),
                                       this->test_x_second_derivative().at(i));
    solver.add_acceleration_lower_bound(
        this->test_t().at(i), this->test_x_second_derivative().at(i) - 0.01);
    solver.add_acceleration_upper_bound(
        this->test_t().at(i), this->test_x_second_derivative().at(i) + 0.01);
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_jerk_constraint(this->test_t().at(i),
                               this->test_x_third_derivative().at(i));
    solver.add_jerk_lower_bound(this->test_t().at(i),
                                this->test_x_third_derivative().at(i) - 0.01);
    solver.add_jerk_upper_bound(this->test_t().at(i),
                                this->test_x_third_derivative().at(i) + 0.01);
  }

  Eigen::VectorXd rst = solver.solve();
  this->assertSolverResult(rst, 0.001);
}

TEST_F(DistanceTimeSolverTest, TestHessianWithDerivatives) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
  Solver::DistanceTimeSolver solver(this->knots(), 3);
  solver.add_distance_point_to_objective(100000, this->test_t(),
                                         this->test_pts());
  solver.add_speed_point_penalty_to_objective(0.0001, this->test_t());
  solver.add_acceleration_point_penalty_to_objective(0.0001, this->test_t());
  solver.add_jerk_point_penalty_to_objective(0.0001, this->test_t());

  Eigen::VectorXd rst = solver.solve();

  this->assertSolverResult(rst, 0.01);
}

TEST_F(DistanceTimeSolverTest, TestSmoothConstraint) {
  // X(t) = 4t^3+3t^2+2t+1 for [0-1]
  // X(t) = 4(t-1)^3 + 15(t-1)^2 + 20(t-1) + 10 for [1-2]
  Solver::DistanceTimeSolver solver(this->knots(), 3);
  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_distance_constraint(this->test_t().at(i), this->test_x().at(i));
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_speed_constraint(this->test_t().at(i),
                                this->test_x_first_derivative().at(i));
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_acceleration_constraint(this->test_t().at(i),
                                       this->test_x_second_derivative().at(i));
  }

  for (int i = 0; i < this->test_t().size(); i++) {
    solver.add_jerk_constraint(this->test_t().at(i),
                               this->test_x_third_derivative().at(i));
  }

  EXPECT_EQ(solver.add_smooth_constraint(3), true);

  Eigen::VectorXd rst = solver.solve();
  this->assertSolverResult(rst, 0.001);
}
