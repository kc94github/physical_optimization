#include "polynomial1d.h"
#include "spline1d.h"
#include <gtest/gtest.h>

TEST(Spline1dTest, BasicTest) {

  // y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
  std::vector<double> coeffs_1 = {1.0, 2.0, 3.0, 4.0};
  // y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
  std::vector<double> coeffs_2 = {10.0, 2.0, 3.0, 4.0};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_2);

  Geometry::Spline1d sp = Geometry::Spline1d(knots, {poly1, poly2});

  std::cout << sp.toString() << std::endl;

  EXPECT_EQ(sp.knots(), knots);
  EXPECT_EQ(sp.size(), 2);
  EXPECT_EQ(sp.order(), 3);
  EXPECT_EQ(sp.param_size(), 8);

  EXPECT_EQ(sp.line_segment(0), poly1);
  EXPECT_EQ(sp.line_segment(1), poly2);

  std::pair<double, double> p1 = {0.0, 1.0};
  std::pair<double, double> p2 = {1.0, 2.0};

  EXPECT_EQ(sp.knot_segment(0), p1);
  EXPECT_EQ(sp.knot_segment(1), p2);
}

TEST(Spline1dTest, PrevKnotIndexSearch) {
  // y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
  std::vector<double> coeffs_1 = {1.0, 2.0, 3.0, 4.0};
  // y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
  std::vector<double> coeffs_2 = {10.0, 2.0, 3.0, 4.0};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_2);

  Geometry::Spline1d sp = Geometry::Spline1d(knots, {poly1, poly2});

  EXPECT_EQ(sp.search_prev_knot_index(0.0), 0);
  EXPECT_EQ(sp.search_prev_knot_index(0.5), 0);
  EXPECT_EQ(sp.search_prev_knot_index(1.0), 0);
  EXPECT_EQ(sp.search_prev_knot_index(1.5), 1);
  EXPECT_EQ(sp.search_prev_knot_index(2.0), 1);
}

TEST(Spline1dTest, EvalTest) {

  // y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
  std::vector<double> coeffs_1 = {1.0, 2.0, 3.0, 4.0};
  // y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
  std::vector<double> coeffs_2 = {10.0, 2.0, 3.0, 4.0};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_2);

  Geometry::Spline1d sp = Geometry::Spline1d(knots, {poly1, poly2});

  EXPECT_EQ(sp.evaluate(2.0), 19.0);
  EXPECT_EQ(sp.derivative(2.0), 20.0);
  EXPECT_EQ(sp.second_derivative(2.0), 30.0);
  EXPECT_EQ(sp.third_derivative(2.0), 24.0);
}

TEST(Spline1dTest, DerivativeTest) {
  // y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
  std::vector<double> coeffs_1 = {1.0, 2.0, 3.0, 4.0};
  // y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
  std::vector<double> coeffs_2 = {10.0, 2.0, 3.0, 4.0};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_2);

  Geometry::Spline1d sp = Geometry::Spline1d(knots, {poly1, poly2});

  std::vector<double> derivative_coeffs_1 = {2, 6, 12};
  std::vector<double> derivative_coeffs_2 = {2, 6, 12};

  EXPECT_EQ(sp.derivative_spline(1).polynomials()[0].coefficients(),
            derivative_coeffs_1);
  EXPECT_EQ(sp.derivative_spline(1).polynomials()[1].coefficients(),
            derivative_coeffs_2);
}

TEST(Spline1dTest, IntegralTest) {
  // y = 4x^3 + 3x^2+ 2x + 1 for [0-1]
  std::vector<double> coeffs_1 = {1.0, 2.0, 3.0, 4.0};
  // y = 4(x-1)^3 + 3(x-1)^2+ 2(x-1) + 10 for [1-2]
  std::vector<double> coeffs_2 = {10.0, 2.0, 3.0, 4.0};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_2);

  Geometry::Spline1d sp = Geometry::Spline1d(knots, {poly1, poly2});

  std::vector<double> integral_coeffs_1 = {0, 1, 1, 1, 1};
  std::vector<double> integral_coeffs_2 = {0, 10, 1, 1, 1};

  EXPECT_EQ(sp.integral_spline(1).polynomials()[0].coefficients(),
            integral_coeffs_1);
  EXPECT_EQ(sp.integral_spline(1).polynomials()[1].coefficients(),
            integral_coeffs_2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}