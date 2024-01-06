#pragma once

#include "abstract.h"
#include "spline1d.h"
#include "polynomial1d.h"

namespace Geometry {

class Spline2d : public Abstract {

    std::vector<double> _knots;
    Spline1d _x_spline;
    Spline1d _y_spline;

public:

    Spline2d(const Spline1d& x_spline, const Spline1d& y_spline):
        _knots(x_spline.knots()),
        _x_spline(x_spline),
        _y_spline(y_spline) {
            assert(x_spline.knots() == y_spline.knots());
            assert(_knots.size() >= 2);
            assert(_x_spline.order() == _y_spline.order());
        }

    static Spline2d spline_from_xy_splines(const Spline1d& x_spline, const Spline1d& y_spline) {
        return Spline2d(x_spline, y_spline);
    }

    static Spline2d spline_from_knots_and_coefficients(const std::vector<double>& knots, const std::vector<std::vector<double>>& x_coefficients_list, const std::vector<std::vector<double>>& y_coefficients_list) {
        assert(x_coefficients_list.size() == y_coefficients_list.size());
        assert(knots.size() - 1 == x_coefficients_list.size());

        std::vector<Polynomial1d> x_poly, y_poly;
        int cur_param_size = -1;
        for (int i=0;i<x_coefficients_list.size();i++) {
            assert(x_coefficients_list[i].size() == y_coefficients_list[i].size());
            if (cur_param_size == -1) cur_param_size = x_coefficients_list[i].size();
            else {
                assert(cur_param_size == x_coefficients_list[i].size());
            }
            x_poly.push_back(Polynomial1d::polynomial_from_coefficients(x_coefficients_list[i]));
            y_poly.push_back(Polynomial1d::polynomial_from_coefficients(y_coefficients_list[i]));
        }
        Spline1d sp_x = Spline1d(knots, x_poly);
        Spline1d sp_y = Spline1d(knots, y_poly);
        return Spline2d(sp_x, sp_y);
    }

    inline std::string toString() const {
        std::string s = std::string("Spline2d: ") + "\n";
        s += _x_spline.toString();
        s += _y_spline.toString();
        return s;
    }

    bool operator==(const Spline2d& other) const {
        if (_x_spline == other._x_spline && _y_spline == other._y_spline) return true;
        return false;
    }

    inline std::vector<double> knots() const {
        return _knots;
    }

    inline std::vector<Polynomial1d> x_polynomials() const {
        return _x_spline.polynomials();
    }

    inline std::vector<Polynomial1d> y_polynomials() const {
        return _y_spline.polynomials();
    }

    inline int order() const {
        return _x_spline.order();
    }

    inline int param_size() const {
        return _x_spline.param_size() + _y_spline.param_size();
    }

    inline int size() const {
        return _x_spline.size();
    }

    inline std::pair<double, double> knot_segment(int index) const {
        assert(index < size());
        return {_knots[index], _knots[index + 1]};
    }

    inline Polynomial1d x_line_segment(const int& index) const {
        assert(index < size());
        return _x_spline.line_segment(index);
    }

    inline Polynomial1d y_line_segment(const int& index) const {
        assert(index < size());
        return _y_spline.line_segment(index);
    } 

    inline int search_prev_knot_index(const double& t) const {
        return _x_spline.search_prev_knot_index(t);
    }

    inline double x_evaluate(const double& t) const {
        return _x_spline.evaluate(t);
    }

    inline double y_evaluate(const double& t) const {
        return _y_spline.evaluate(t);
    }

    inline double x_derivative(const double& t) const {
        return _x_spline.derivative(t);
    }

    inline double y_derivative(const double& t) const {
        return _y_spline.derivative(t);
    }

    inline double x_second_derivative(const double& t) const {
        return _x_spline.second_derivative(t);
    }

    inline double y_second_derivative(const double& t) const {
        return _y_spline.second_derivative(t);
    }

    inline double x_third_derivative(const double& t) const {
        return _x_spline.third_derivative(t);
    }

    inline double y_third_derivative(const double& t) const {
        return _y_spline.third_derivative(t);
    }

    Spline2d derivative_spline(const int& order = 1) const {
        Spline1d x_derivative_spline = _x_spline.derivative_spline(order);
        Spline1d y_derivative_spline = _y_spline.derivative_spline(order);
        return Spline2d(x_derivative_spline, y_derivative_spline);
    }

    Spline2d integral_spline(const int& order = 1) const {
        Spline1d x_integral_spline = _x_spline.integral_spline(order);
        Spline1d y_integral_spline = _y_spline.integral_spline(order);
        return Spline2d(x_integral_spline, y_integral_spline);
    }

};

}
