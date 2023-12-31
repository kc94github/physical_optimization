#include "polynomial1d.h"
#include "polynomial2d.h"
#include <gtest/gtest.h>

TEST(Polynomial2dTest, BasicTest) {

  std::vector<double> x_coeffs = {4, 3, 2, 1};
  std::vector<double> y_coeffs = {9, 8, 7, 6};

  Geometry::Polynomial2d a =
      Geometry::Polynomial2d::polynomial_from_coefficients(x_coeffs, y_coeffs);

  EXPECT_EQ(a.order(), 3);
  EXPECT_EQ(a.param_size(), 8);
  EXPECT_EQ(a.x_coefficients(), x_coeffs);
  EXPECT_EQ(a.y_coefficients(), y_coeffs);

  Geometry::Polynomial2d b =
      Geometry::Polynomial2d::polynomial_from_coefficients(x_coeffs, y_coeffs);
  EXPECT_EQ(a, b);
}

TEST(Polynomial2dTest, BasicTest2) {

  std::vector<double> x_coeffs = {4, 3, 2, 1};
  std::vector<double> y_coeffs = {9, 8, 7, 6};

  Geometry::Polynomial1d a =
      Geometry::Polynomial1d::polynomial_from_coefficients(x_coeffs);
  Geometry::Polynomial1d b =
      Geometry::Polynomial1d::polynomial_from_coefficients(y_coeffs);

  Geometry::Polynomial2d c =
      Geometry::Polynomial2d::polynomial_from_1d_polys(a, b);

  EXPECT_EQ(c.order(), 3);
  EXPECT_EQ(c.param_size(), 8);
  EXPECT_EQ(c.x_coefficients(), x_coeffs);
  EXPECT_EQ(c.y_coefficients(), y_coeffs);
}

TEST(Polynomial2dTest, EvalTest) {

  std::vector<double> x_coeffs = {4, 3, 2, 1};
  std::vector<double> y_coeffs = {9, 8, 7, 6};

  Geometry::Polynomial2d a =
      Geometry::Polynomial2d::polynomial_from_coefficients(x_coeffs, y_coeffs);

  EXPECT_EQ(a.evaluate_x(2.0), 26.0);
  EXPECT_EQ(a.derivative_x(2.0), 23.0);
  EXPECT_EQ(a.second_derivative_x(2.0), 16.0);
  EXPECT_EQ(a.third_derivative_x(2.0), 6.0);

  EXPECT_EQ(a.evaluate_y(2.0), 101.0);
  EXPECT_EQ(a.derivative_y(2.0), 108.0);
  EXPECT_EQ(a.second_derivative_y(2.0), 86.0);
  EXPECT_EQ(a.third_derivative_y(2.0), 36.0);
}

TEST(PolynomialTest, DerivativeTest) {
  std::vector<double> x_coeffs = {4, 3, 2, 1};
  std::vector<double> y_coeffs = {9, 8, 7, 6};

  Geometry::Polynomial2d a =
      Geometry::Polynomial2d::polynomial_from_coefficients(x_coeffs, y_coeffs);

  std::vector<double> derivative_x_coeffs = {3, 4, 3};
  std::vector<double> derivative_y_coeffs = {8, 14, 18};

  Geometry::Polynomial2d c = a.derivative_polynomial(1);
  EXPECT_EQ(c.x_coefficients(), derivative_x_coeffs);
  EXPECT_EQ(c.y_coefficients(), derivative_y_coeffs);
}

TEST(PolynomialTest, IntegralTest) {
  std::vector<double> x_coeffs = {4, 3, 0, 1};
  std::vector<double> y_coeffs = {9, 8, 0, 6};
  Geometry::Polynomial2d a =
      Geometry::Polynomial2d::polynomial_from_coefficients(x_coeffs, y_coeffs);

  std::vector<double> integral_coeffs_x = {0, 4, 1.5, 0, 0.25};
  std::vector<double> integral_coeffs_y = {0, 9, 4, 0, 1.5};

  Geometry::Polynomial2d c = a.integral_polynomial(1);
  EXPECT_EQ(c.x_coefficients(), integral_coeffs_x);
  EXPECT_EQ(c.y_coefficients(), integral_coeffs_y);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}