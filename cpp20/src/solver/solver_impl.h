#include <vector>
#include <iostream>
#include <string>
#include <cassert>

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Solver {

class SolverImpl {

    private:

        uint _size;
        Eigen::SparseMatrix<double> _hessian_matrix;
        
        Eigen::VectorXd _gradient_vector;

        Eigen::SparseMatrix<double> _total_constraint_matrix;
        Eigen::VectorXd _total_constraint_upper_bound;
        Eigen::VectorXd _total_constraint_lower_bound;

    public:

        SolverImpl(uint size = 1):_size(size) {
            reset_matrices();
        }

        inline void reset_matrices(uint size = 0) {
            if (size != 0) _size = size;

            _hessian_matrix.resize(_size, _size);
            _hessian_matrix.setZero();

            _gradient_vector.resize(_size);
            _gradient_vector.setZero();

            _total_constraint_matrix.resize(0, _size);
            _total_constraint_upper_bound.resize(0);
            _total_constraint_lower_bound.resize(0);
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

        inline auto constraint_matrix() const {
            return _total_constraint_matrix;
        }

        inline auto upper_bound_vector() const {
            return _total_constraint_upper_bound;
        }

        inline auto lower_bound_vector() const {
            return _total_constraint_lower_bound;
        }

        bool add_value_to_gradient_vector(const uint row_index, const double value);

        bool add_to_objective_function_from_sparse(const uint start_row, const uint start_col, const uint add_row_size, const uint add_col_size, const Eigen::SparseMatrix<double>& add_hessian_submatrix, const Eigen::VectorXd& add_gradient_subvector = Eigen::Vector<double, 0>::Zero());

        bool set_objective_function(const Eigen::MatrixXd& hessian_matrix, const Eigen::VectorXd& gradient_subvector = Eigen::Vector<double, 0>::Zero());

        bool set_objective_function_from_sparse(const Eigen::SparseMatrix<double>& hessian_submatrix, const Eigen::VectorXd& gradient_subvector = Eigen::Vector<double, 0>::Zero());

        bool set_upper_bound(const Eigen::VectorXd &upper_bound);

        bool set_lower_bound(const Eigen::VectorXd &lower_bound);

        bool add_equality_constraint(const Eigen::MatrixXd &equality_matrix, const Eigen::VectorXd &equality_vector);

        bool add_equality_constraint_with_sparse(const Eigen::SparseMatrix<double> &equality_matrix, const Eigen::VectorXd &equality_vector);

        bool add_inequality_constraint_with_sparse(const Eigen::SparseMatrix<double> &inequality_matrix, const Eigen::VectorXd &inequality_vector);

        bool add_inequality_constraint(const Eigen::MatrixXd &inequality_matrix, const Eigen::VectorXd &inequality_vector);

        // bool _add_constraint_helper(const Eigen::SparseMatrix<double> &coefficient_matrix, const Eigen::VectorXd &constant_vector, Eigen::SparseMatrix<double> &constraint_matrix, Eigen::VectorXd &constraint_vector, uint size);

        bool add_constraint_with_bounds(const Eigen::SparseMatrix<double> &constraint_matrix, const Eigen::VectorXd &upper_bound, const Eigen::VectorXd &lower_bound);

        Eigen::VectorXd solve();
};      

}