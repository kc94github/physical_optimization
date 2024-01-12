#pragma once

#include <vector>
#include "abstract.h"
#include "solver_impl.h"
#include "spline1d.h"
#include "coefficient_base.h"

namespace Solver {

class SplineNdSolver : public Geometry::CoefficientBase {

    std::vector<double> _knots;
    uint _spline_order;
    uint _dimension;
    uint _param_size;

    Solver::SolverImpl _impl;

public:

    SplineNdSolver(const std::vector<double>& knots, const uint& spline_order, const uint& dimension = 1);
    

    std::string toString() const;

    Solver::SolverImpl& solver() {
        return _impl;
    }

    inline uint size() const {
        return _impl.size();
    }

    inline auto hessian_matrix() const {
        return _impl.hessian_matrix();
    }

    inline auto gradient_vector() const {
        return _impl.gradient_vector();
    }

    inline Eigen::VectorXd solve() {
        return _impl.solve();
    }

    inline uint dimension() const {
        return _dimension;
    }

    inline uint param_size() const {
        return _param_size;
    }

    void add_regularization(const double& regularization_param);

    template<typename EvaluateFunc>
    std::pair<uint, std::vector<double> > get_knot_index_and_coefficient(EvaluateFunc coefficient_func, const double& t) const{
        uint index = _search_prev_knot_index(t);
        double relative_time = t - _knots[index];
        return {index, (this->*coefficient_func)(relative_time)};
    }

    template<typename KnotIndexAndCoefficientFunc>
    bool apply_equality_constraint(KnotIndexAndCoefficientFunc func, const double& t, const std::vector<double>& point) {
        std::pair<uint, std::vector<double> > p = func(t);
        Eigen::MatrixXd matrix_A = Eigen::MatrixXd::Zero(_dimension, _param_size);

        Eigen::MatrixXd coeff(1, _spline_order + 1);
        for (int i=0;i<p.second.size();i++) {
            coeff(1, i) = p.second.at(i);
        }

        uint start_index = p.first * (_dimension * (_spline_order + 1));
        for (int i=0;i<_dimension;i++) {
            matrix_A.block(i, start_index, 1, _spline_order + 1) = coeff;
            start_index += (_spline_order + 1);
        }

        assert(point.size() == _dimension);
        Eigen::VectorXd matrix_B = Eigen::VectorXd::Zero(_dimension);
        for (int i=0;i<_dimension;i++) {
            matrix_B(i) = point.at(i);
        }
        return _impl.add_equality_constraint(matrix_A, matrix_B);
    }

    template<typename KnotIndexAndCoefficientFunc>
    bool apply_inequality_constraint(KnotIndexAndCoefficientFunc func, const double& t, const std::vector<double>& point) {
        std::pair<uint, std::vector<double> > p = func(t);
        Eigen::MatrixXd matrix_A = Eigen::MatrixXd::Zero(_dimension, _param_size);

        Eigen::MatrixXd coeff(1, _spline_order + 1);
        for (int i=0;i<p.second.size();i++) {
            coeff(1, i) = p.second.at(i);
        }

        uint start_index = p.first * (_dimension * (_spline_order + 1));
        for (int i=0;i<_dimension;i++) {
            matrix_A.block(i, start_index, 1, _spline_order + 1) = coeff;
            start_index += (_spline_order + 1);
        }

        assert(point.size() == _dimension);
        Eigen::VectorXd matrix_B = Eigen::VectorXd::Zero(_dimension);
        for (int i=0;i<_dimension;i++) {
            matrix_B(i) = point.at(i);
        }
        return _impl.add_inequality_constraint(matrix_A, matrix_B);
    }

    inline bool add_smooth_constraint(const uint& order=0) {
        assert(order <= 3);
        bool cur = _add_smooth_constraint_order_zero();
        if (order == 0) return cur;
        cur &= _add_smooth_constraint_order_one();
        if (order == 1) return cur;
        cur &= _add_smooth_constraint_order_two();
        if (order == 2) return cur;
        cur &= _add_smooth_constraint_order_three();
        return cur;
    }

    template<typename EvaluateFunc>
    bool apply_derivative_to_objective(EvaluateFunc coefficient_func, const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& point_ref) {
        assert(t_ref.size() == point_ref.size());
        if (t_ref.size() == 0) return false;
        // J = 0.5 * X_tran * Hessian * X + X_tran * Gradient
        uint param_number = _spline_order + 1;
        for (uint i=0;i<t_ref.size();i++) {
            std::vector<double> cur_pt = point_ref[i];
            assert(cur_pt.size() == _dimension);
            uint cur_index = _search_prev_knot_index(t_ref[i]);
            double cur_relative_time = t_ref[i] - _knots[cur_index];
            uint cur_start_row = _dimension * cur_index * param_number;

            std::vector<double> cur_gradient(_dimension);
            for (uint d=0;d<_dimension;d++) {
                cur_gradient.push_back(-2.0 * cur_pt[d] * weight);
            }

            for (uint p=0;p<param_number;p++) {
                for (uint d=0;d<_dimension;d++) {
                    uint idx = cur_start_row + d * param_number;
                    _impl.add_value_to_gradient_vector(p + idx, cur_gradient[d]);
                    cur_gradient[d] *= cur_relative_time;
                }
            }

            Eigen::MatrixXd cur_hessian_matrix = Eigen::MatrixXd::Zero(param_number, param_number);
            std::vector<double> cur_coefficient = (this->*coefficient_func)(cur_relative_time);

            for (uint j=0;j<param_number;j++) {
                for (uint k=0;k<param_number;k++) {
                    cur_hessian_matrix(j, k) = 2.0 * cur_coefficient[j] * cur_coefficient[k];
                }
            }

            for (uint d=0;d<_dimension;d++) {
                uint idx = cur_start_row + d * param_number;
                _impl.add_to_objective_function(idx, idx, param_number, param_number, weight * cur_hessian_matrix);
            }
        }
        return true;
    }

    bool add_point_constraint(const double& t, const std::vector<double>& point);

    bool add_point_first_derivative_constraint(const double& t, const std::vector<double>& point);

    bool add_point_second_derivative_constraint(const double& t, const std::vector<double>& point);

    bool add_point_third_derivative_constraint(const double& t, const std::vector<double>& point);

    bool add_reference_points_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& point_ref);

    bool add_first_derivative_points_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& point_ref);
    
    bool add_second_derivative_points_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& point_ref);

    bool add_third_derivative_points_to_objective(const double& weight, const std::vector<double>& t_ref, const std::vector<std::vector<double>>& point_ref);


private:

    inline uint _search_prev_knot_index(double t) const {
        return Geometry::Spline1d::static_search_prev_knot_index(_knots, t);
    }

    bool _add_smooth_constraint_order_zero();

    bool _add_smooth_constraint_order_one();

    bool _add_smooth_constraint_order_two();

    bool _add_smooth_constraint_order_three();

};

}