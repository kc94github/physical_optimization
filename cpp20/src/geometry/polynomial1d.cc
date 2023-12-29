#include "polynomial1d.h"

using namespace Geometry;

double Polynomial1d::evaluate(const double &t) {
  return summation(
      std::bind(&CoefficientBase::t_coefficient, this, std::placeholders::_1),
      t);
}

double Polynomial1d::derivative(const double &t) {
  return summation(std::bind(&CoefficientBase::t_first_derivative_coefficient,
                             this, std::placeholders::_1),
                   t);
}

double Polynomial1d::second_derivative(const double &t) {
  return summation(std::bind(&CoefficientBase::t_second_derivative_coefficient,
                             this, std::placeholders::_1),
                   t);
}

double Polynomial1d::third_derivative(const double &t) {
  return summation(std::bind(&CoefficientBase::t_third_derivative_coefficient,
                             this, std::placeholders::_1),
                   t);
}

Polynomial1d Polynomial1d::derivative_polynomial(const uint &order) {
  std::vector<double> old_coefficients = _coefficients;
  for (int t = 0; t < order; t++) {
    std::vector<double> new_coefficients;
    for (int i = 1; i < old_coefficients.size(); i++) {
      new_coefficients.push_back(old_coefficients[i] * (double)i);
    }
    old_coefficients = new_coefficients;
  }
  return Polynomial1d::polynomial_from_coefficients(old_coefficients);
}

Polynomial1d Polynomial1d::integral_polynomial(const uint &order) {
  std::vector<double> old_coefficients = _coefficients;
  for (int t = 0; t < order; t++) {
    std::vector<double> new_coefficients = {0};
    for (int i = 0; i < old_coefficients.size(); i++) {
      new_coefficients.push_back(old_coefficients[i] / (double)(i + 1));
    }
    old_coefficients = new_coefficients;
  }
  return Polynomial1d::polynomial_from_coefficients(old_coefficients);
}