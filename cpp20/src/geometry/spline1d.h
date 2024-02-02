#pragma once

#include <vector>
#include "abstract.h"
#include "polynomial1d.h"

namespace Geometry {

class Spline1d : public Abstract {

    std::vector<double> _knots;
    std::vector<Polynomial1d> _polynomials;

public:

    Spline1d(std::vector<double> knots, const std::vector<Polynomial1d>& polynomials):
        _knots(knots),
        _polynomials(polynomials) {
            assert(knots.size() == polynomials.size()+1);
            assert(knots.size() >= 2);
        }

    inline std::string toString() const {
        std::string s = "Spline1d with knot size: " + std::to_string(_knots.size()) + "\n";
        for (const auto& poly : _polynomials) {
            s += poly.toString() + "\n";
        }
        return s;
    }

    static int static_search_prev_knot_index(const std::vector<double>& knots, const double& t) {
        assert(t >= knots[0] && t <= knots.back());
        auto it = std::lower_bound(knots.begin(), knots.end(), t);
        int distance = it - knots.begin();
        if (distance > 0) return distance - 1;
        else return distance;
    }

    bool operator==(const Spline1d& other) const {
        if (_knots.size() != other._knots.size()) return false;
        for (int i=0;i<_polynomials.size();i++) {
            if (_polynomials[i] != other._polynomials[i]) return false;
        }
        return true;
    }

    inline std::vector<double> knots() const {
        return _knots;
    }

    inline std::vector<Polynomial1d> polynomials() const {
        return _polynomials;
    }

    inline int order() const {
        return _polynomials[0].order();
    }

    inline int param_size() const {
        return _polynomials[0].param_size() * _polynomials.size();
    }

    inline int size() const {
        return _polynomials.size();
    }

    inline std::pair<double, double> knot_segment(int index) const {
        assert(index < size());
        return {_knots[index], _knots[index + 1]};
    }

    inline Polynomial1d line_segment(const int& index) const {
        assert(index < size());
        return _polynomials[index];
    }

    inline int search_prev_knot_index(const double& t) const {
        return Spline1d::static_search_prev_knot_index(_knots, t);
    }


    template<typename EvaluateFunc>
    double spline_relative_eval(EvaluateFunc coefficient_func, const double& t) const{
        int index = search_prev_knot_index(t);
        double relative_time = t - _knots[index];
        return (_polynomials[index].*coefficient_func)(relative_time);
    }

    double evaluate(const double& t) const;

    double derivative(const double& t) const;

    double second_derivative(const double& t) const;

    double third_derivative(const double& t) const;

    Spline1d derivative_spline(const int& order = 1) const;

    Spline1d integral_spline(const int& order = 1) const;

};

}