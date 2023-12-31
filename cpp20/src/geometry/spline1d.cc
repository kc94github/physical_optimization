#include "spline1d.h"

using namespace Geometry;

double Spline1d::evaluate(const double& t) const {
    return spline_relative_eval(&Polynomial1d::evaluate, t);
}

// double Spline1d::evaluate(const double& t) const {
//     int index = _search_prev_knot_index(t);
//     return spline_relative_eval(std::bind(&Polynomial1d::evaluate,
//                             &_polynomials[index], std::placeholders::_1), t, index);
// }

double Spline1d::derivative(const double& t) const {
    return spline_relative_eval(&Polynomial1d::derivative, t);
}

double Spline1d::second_derivative(const double& t) const {
    return spline_relative_eval(&Polynomial1d::second_derivative, t);
}

double Spline1d::third_derivative(const double& t) const {
    return spline_relative_eval(&Polynomial1d::third_derivative, t);
}

Spline1d Spline1d::derivative_spline(const int& order) const {
    std::vector<Polynomial1d> derivative_polys;
    for (int i=0;i<_polynomials.size();i++) {
        derivative_polys.emplace_back(_polynomials[i].derivative_polynomial(order));
    }
    return Spline1d(_knots, derivative_polys);
}

Spline1d Spline1d::integral_spline(const int& order) const {
    std::vector<Polynomial1d> integral_polys;
    for (int i=0;i<_polynomials.size();i++) {
        integral_polys.emplace_back(_polynomials[i].integral_polynomial(order));
    }
    return Spline1d(_knots, integral_polys);
}