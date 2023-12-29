#include "polynomial1d.h"
#include <gtest/gtest.h>

TEST(PolynomialTest, BasicTest) {

  std::vector<double> coeffs = {4,3,2,1};
  Geometry::Polynomial1d a = Geometry::Polynomial1d::polynomial_from_coefficients(coeffs);
  EXPECT_EQ(a.order(), 3);
  EXPECT_EQ(a.param_size(), 4);
  EXPECT_EQ(a.coefficients(), coeffs);

  Geometry::Polynomial1d b = Geometry::Polynomial1d::polynomial_from_coefficients(coeffs);
    EXPECT_EQ(a, b);
}

TEST(PolynomialTest, EvalTest) {

  std::vector<double> coeffs = {4,3,2,1};
  Geometry::Polynomial1d a = Geometry::Polynomial1d::polynomial_from_coefficients(coeffs);
  EXPECT_EQ(a.evaluate(2.0), 26.0);
  EXPECT_EQ(a.derivative(2.0), 23.0);
  EXPECT_EQ(a.second_derivative(2.0), 16.0);
  EXPECT_EQ(a.third_derivative(2.0), 6.0);
}

TEST(PolynomialTest, DerivativeTest) {
  std::vector<double> coeffs = {4,3,2,1};
  Geometry::Polynomial1d a = Geometry::Polynomial1d::polynomial_from_coefficients(coeffs);
  std::vector<double> derivative_coeffs = {3,4,3};
  EXPECT_EQ(a.derivative_polynomial(1).coefficients(), derivative_coeffs);
}

TEST(PolynomialTest, IntegralTest) {
  std::vector<double> coeffs = {4,3,0,1};
  Geometry::Polynomial1d a = Geometry::Polynomial1d::polynomial_from_coefficients(coeffs);
  std::vector<double> integral_coeffs = {0,4,1.5,0,0.25};
  EXPECT_EQ(a.integral_polynomial(1).coefficients(), integral_coeffs);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}