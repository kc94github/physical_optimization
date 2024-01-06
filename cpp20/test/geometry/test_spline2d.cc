#include "polynomial1d.h"
#include "spline1d.h"
#include "spline2d.h"
#include <gtest/gtest.h>

class Spline2dTest : public ::testing::Test {

protected:
  static std::vector<double> knots() { return {0.0, 1.0, 2.0}; }

  static Geometry::Polynomial1d poly1() {
    // y = 4t^3 + 3t^2+ 2t + 1 for [0-1]
    std::vector<double> coeffs_1 = {1.0, 2.0, 3.0, 4.0};
    Geometry::Polynomial1d poly1 =
        Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_1);
    return poly1;
  }

  static Geometry::Polynomial1d poly2() {
    // y = 4(t-1)^3 + 3(t-1)^2+ 2(t-1) + 10 for [1-2]
    std::vector<double> coeffs_2 = {10.0, 2.0, 3.0, 4.0};
    Geometry::Polynomial1d poly2 =
        Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_2);
    return poly2;
  }

  static Geometry::Polynomial1d poly3() {
    // y = 5t^3 + 2 for [0-1]
    std::vector<double> coeffs_3 = {2.0, 0.0, 0.0, 5.0};
    Geometry::Polynomial1d poly3 =
        Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_3);
    return poly3;
  }

  static Geometry::Polynomial1d poly4() {
    // y = 5(t-1)^3 + 7 for [1-2]
    std::vector<double> coeffs_4 = {7.0, 0.0, 0.0, 5.0};
    Geometry::Polynomial1d poly4 =
        Geometry::Polynomial1d::polynomial_from_coefficients(coeffs_4);
    return poly4;
  }

  static Geometry::Spline2d Get2dSplineExample() {
    std::vector<double> knots = {0.0, 1.0, 2.0};
    Geometry::Polynomial1d poly1 = Spline2dTest::poly1();
    Geometry::Polynomial1d poly2 = Spline2dTest::poly2();
    Geometry::Spline1d sp_1 = Geometry::Spline1d(knots, {poly1, poly2});

    Geometry::Polynomial1d poly3 = Spline2dTest::poly3();
    Geometry::Polynomial1d poly4 = Spline2dTest::poly4();
    Geometry::Spline1d sp_2 = Geometry::Spline1d(knots, {poly3, poly4});

    Geometry::Spline2d sp =
        Geometry::Spline2d::spline_from_xy_splines(sp_1, sp_2);

    return sp;
  }
};

TEST_F(Spline2dTest, BasicTest) {

  Geometry::Spline2d sp = Spline2dTest::Get2dSplineExample();

  EXPECT_EQ(sp.knots(), Spline2dTest::knots());
  EXPECT_EQ(sp.size(), 2);
  EXPECT_EQ(sp.order(), 3);
  EXPECT_EQ(sp.param_size(), 16);

  EXPECT_EQ(sp.x_line_segment(0), Spline2dTest::poly1());
  EXPECT_EQ(sp.x_line_segment(1), Spline2dTest::poly2());

  EXPECT_EQ(sp.y_line_segment(0), Spline2dTest::poly3());
  EXPECT_EQ(sp.y_line_segment(1), Spline2dTest::poly4());

  std::pair<double, double> p1 = {0.0, 1.0};
  std::pair<double, double> p2 = {1.0, 2.0};

  EXPECT_EQ(sp.knot_segment(0), p1);
  EXPECT_EQ(sp.knot_segment(1), p2);
}

TEST_F(Spline2dTest, PrevKnotIndexSearch) {
  Geometry::Spline2d sp = Spline2dTest::Get2dSplineExample();

  EXPECT_EQ(sp.search_prev_knot_index(0.0), 0);
  EXPECT_EQ(sp.search_prev_knot_index(0.5), 0);
  EXPECT_EQ(sp.search_prev_knot_index(1.0), 0);
  EXPECT_EQ(sp.search_prev_knot_index(1.5), 1);
  EXPECT_EQ(sp.search_prev_knot_index(2.0), 1);
}

TEST_F(Spline2dTest, EvalTest) {

  Geometry::Spline2d sp = Spline2dTest::Get2dSplineExample();

  EXPECT_EQ(sp.x_evaluate(2.0), 19.0);
  EXPECT_EQ(sp.x_derivative(2.0), 20.0);
  EXPECT_EQ(sp.x_second_derivative(2.0), 30.0);
  EXPECT_EQ(sp.x_third_derivative(2.0), 24.0);

  EXPECT_EQ(sp.y_evaluate(2.0), 12.0);
  EXPECT_EQ(sp.y_derivative(2.0), 15.0);
  EXPECT_EQ(sp.y_second_derivative(2.0), 30.0);
  EXPECT_EQ(sp.y_third_derivative(2.0), 30.0);
}

TEST_F(Spline2dTest, DerivativeTest) {
  Geometry::Spline2d sp = Spline2dTest::Get2dSplineExample();

  std::vector<double> derivative_coeffs_1 = {2, 6, 12};
  std::vector<double> derivative_coeffs_2 = {2, 6, 12};

  std::vector<double> derivative_coeffs_3 = {0, 0, 15};
  std::vector<double> derivative_coeffs_4 = {0, 0, 15};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(derivative_coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(derivative_coeffs_2);
  Geometry::Spline1d sp_1 = Geometry::Spline1d(knots, {poly1, poly2});

  Geometry::Polynomial1d poly3 =
      Geometry::Polynomial1d::polynomial_from_coefficients(derivative_coeffs_3);
  Geometry::Polynomial1d poly4 =
      Geometry::Polynomial1d::polynomial_from_coefficients(derivative_coeffs_4);
  Geometry::Spline1d sp_2 = Geometry::Spline1d(knots, {poly3, poly4});

  Geometry::Spline2d de_sp =
      Geometry::Spline2d::spline_from_xy_splines(sp_1, sp_2);

  EXPECT_EQ(sp.derivative_spline(1), de_sp);
}

TEST_F(Spline2dTest, IntegralTest) {
  Geometry::Spline2d sp = Spline2dTest::Get2dSplineExample();

  std::vector<double> integral_coeffs_1 = {0, 1, 1, 1, 1};
  std::vector<double> integral_coeffs_2 = {0, 10, 1, 1, 1};

  std::vector<double> integral_coeffs_3 = {0, 2, 0, 0, 1.25};
  std::vector<double> integral_coeffs_4 = {0, 7, 0, 0, 1.25};

  std::vector<double> knots = {0.0, 1.0, 2.0};
  Geometry::Polynomial1d poly1 =
      Geometry::Polynomial1d::polynomial_from_coefficients(integral_coeffs_1);
  Geometry::Polynomial1d poly2 =
      Geometry::Polynomial1d::polynomial_from_coefficients(integral_coeffs_2);
  Geometry::Spline1d sp_1 = Geometry::Spline1d(knots, {poly1, poly2});

  Geometry::Polynomial1d poly3 =
      Geometry::Polynomial1d::polynomial_from_coefficients(integral_coeffs_3);
  Geometry::Polynomial1d poly4 =
      Geometry::Polynomial1d::polynomial_from_coefficients(integral_coeffs_4);
  Geometry::Spline1d sp_2 = Geometry::Spline1d(knots, {poly3, poly4});

  Geometry::Spline2d in_sp =
      Geometry::Spline2d::spline_from_xy_splines(sp_1, sp_2);

  EXPECT_EQ(sp.integral_spline(1), in_sp);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}