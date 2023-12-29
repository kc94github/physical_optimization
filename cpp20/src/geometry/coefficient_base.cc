#include "coefficient_base.h"

using namespace Geometry;


std::vector<double> CoefficientBase::t_coefficient(double t) const {
  std::vector<double> res(_order + 1, 0);
  res[0] = 1;
  for (int i = 1; i <= _order; ++i) {
    res[i] = res[i - 1] * t;
  }
  return res;
}

std::vector<double>
CoefficientBase::t_first_derivative_coefficient(double t) const {
  std::vector<double> res(_order + 1, 0);
  auto t_coeff = t_coefficient(t);
  for (int i = 1; i <= _order; ++i) {
    res[i] = t_coeff[i - 1] * i;
  }
  return res;
}

std::vector<double>
CoefficientBase::t_second_derivative_coefficient(double t) const {
  std::vector<double> res(_order + 1, 0);
  auto t_first_derivative_coeff = t_first_derivative_coefficient(t);
  for (int i = 2; i <= _order; ++i) {
    res[i] = t_first_derivative_coeff[i - 1] * i;
  }
  return res;
}

std::vector<double>
CoefficientBase::t_third_derivative_coefficient(double t) const {
  std::vector<double> res(_order + 1, 0);
  auto t_second_derivative_coeff = t_second_derivative_coefficient(t);
  for (int i = 3; i <= _order; ++i) {
    res[i] = t_second_derivative_coeff[i - 1] * i;
  }
  return res;
}