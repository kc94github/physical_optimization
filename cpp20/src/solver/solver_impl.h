#include <vector>
#include <iostream>
#include <string>
#include <cassert>

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>

namespace Solver {

class SolverImpl {

    private:

        uint _size;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _hessian_matrix;
        Eigen::Vector<double, Eigen::Dynamic> _gradient_vector;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _equality_constraint_matrix;
        Eigen::Vector<double, Eigen::Dynamic> _equality_constraint_vector;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _inequality_constraint_matrix;
        Eigen::Vector<double, Eigen::Dynamic> _inequality_constraint_vector;

        // Use 0 size to distingulish with-bound and without-bound problem
        Eigen::Vector<double, Eigen::Dynamic> _lower_bound_vector;
        Eigen::Vector<double, Eigen::Dynamic> _upper_bound_vector;

        inline void reset_matrices() {
            // _hessian_matrix = Eigen::Matrix<double, _size, _size>::Zero();
            // _gradient_vector = Eigen::Vector<double, _size>::Zero();

            _hessian_matrix.resize(_size, _size);
            _hessian_matrix.setZero();

            _gradient_vector.resize(_size);
            _gradient_vector.setZero();

            _equality_constraint_matrix.resize(0, _size);
            _equality_constraint_vector.resize(0);

            _inequality_constraint_matrix.resize(0, _size);
            _inequality_constraint_vector.resize(0);

            _lower_bound_vector.resize(0);
            _upper_bound_vector.resize(0);
        }

    public:

        SolverImpl(uint size = 1):_size(size) {
            reset_matrices();
        }

        inline std::string toString() const {
            std::stringstream ss_hessian, ss_gradient;
            ss_hessian << _hessian_matrix;
            ss_gradient << _gradient_vector;

            return "SolverImpl with size: " + std::to_string(_size) + \
            "\nHessian Matrix: " + ss_hessian.str() + \
            "\nGradient Vector: " + ss_gradient.str();
        }

        inline uint size() const {
            return _size;
        }

        inline auto hessian_matrix() const {
            return _hessian_matrix;
        }

        inline auto gradient_vector() const {
            return _gradient_vector;
        }

        bool add_value_to_gradient_vector(const uint row_index, const double value);

        bool add_to_objective_function(const uint start_row, const uint start_col, const uint add_row_size, const uint add_col_size, const Eigen::MatrixXd& add_hessian_submatrix, const Eigen::VectorXd& add_gradient_subvector = Eigen::Vector<double, 0>::Zero());

        bool set_objective_function(const Eigen::MatrixXd& hessian_submatrix, const Eigen::VectorXd& gradient_subvector = Eigen::Vector<double, 0>::Zero());

};

}