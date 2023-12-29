#pragma once

#include <vector>
#include "abstract.h"
#include "polynomial1d.h"

namespace Geometry {

class Polynomial2d : public Abstract {

    int _order;
    int _param_size;
    Polynomial1d _x_polynomial;
    Polynomial1d _y_polynomial;

public:

    Polynomial2d(const std::vector<double>& x_coeffs, const std::vector<double>& y_coeffs):
        _order(x_coeffs.size()-1),
        _param_size(x_coeffs.size() + y_coeffs.size()),
        _x_polynomial(Polynomial1d(x_coeffs)),
        _y_polynomial(Polynomial1d(y_coeffs)) {
        
        assert(x_coeffs.size() == y_coeffs.size());
    }

    Polynomial2d(const Polynomial1d& x_polynomial, const Polynomial1d& y_polynomial):
        _order(x_polynomial.order()),
        _param_size(x_polynomial.param_size() + y_polynomial.param_size()),
        _x_polynomial(x_polynomial),
        _y_polynomial(y_polynomial) {
        assert(x_polynomial.order() == y_polynomial.order());

    }

    static Polynomial2d polynomial_from_coefficients(const std::vector<double>& x_coeffs, const std::vector<double>& y_coeffs) {
        return Polynomial2d(x_coeffs, y_coeffs);
    }

    static Polynomial2d polynomial_from_1d_polys(const Polynomial1d& x_polynomial, const Polynomial1d& y_polynomial) {
        return Polynomial2d(x_polynomial, y_polynomial);
    }

    inline std::string toString() const {
        return "Polynomial2d with order: " + std::to_string(_order);
    }

    inline int order() const {
        return _order;
    }

    inline int param_size() const {
        return _param_size;
    }

    inline std::vector<double> x_coefficients() const {
        return _x_polynomial.coefficients();
    }

    inline std::vector<double> y_coefficients() const {
        return _y_polynomial.coefficients();
    }

    inline Polynomial1d x_polynomial() const {
        return _x_polynomial;
    }

    inline Polynomial1d y_polynomial() const {
        return _y_polynomial;
    }

    bool operator==(const Polynomial2d& other) const {
        return x_coefficients() == other.x_coefficients() && y_coefficients() == other.y_coefficients(); 
    }

    inline double evaluate_x(const double& t) const {
        return _x_polynomial.evaluate(t);
    }

    inline double evaluate_y(const double& t) const {
        return _y_polynomial.evaluate(t);
    }

    inline double derivative_x(const double& t) const {
        return _x_polynomial.derivative(t);
    }

    inline double derivative_y(const double& t) const {
        return _y_polynomial.derivative(t);
    }

    inline double second_derivative_x(const double& t) const {
        return _x_polynomial.second_derivative(t);
    }

    inline double second_derivative_y(const double& t) const {
        return _y_polynomial.second_derivative(t);
    }

    inline double third_derivative_x(const double& t) const {
        return _x_polynomial.third_derivative(t);
    }

    inline double third_derivative_y(const double& t) const {
        return _y_polynomial.third_derivative(t);
    }

    inline Polynomial2d derivative_polynomial(const int& order = 1) const {
        Polynomial1d x_de = _x_polynomial.derivative_polynomial(order);
        Polynomial1d y_de = _y_polynomial.derivative_polynomial(order);
        return Polynomial2d::polynomial_from_1d_polys(x_de, y_de);
    }

    inline Polynomial2d integral_polynomial(const int& order = 1) const {
        Polynomial1d x_in = _x_polynomial.integral_polynomial(order);
        Polynomial1d y_in = _y_polynomial.integral_polynomial(order);
        return Polynomial2d::polynomial_from_1d_polys(x_in, y_in);
    }




};

}