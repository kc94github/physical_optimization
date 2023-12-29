#include <vector>
#include <iostream>
#include <string>

#pragma once

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include "abstract.h"

namespace Geometry {

// CoefficientBase class
class CoefficientBase : public Abstract {
    private:
        int _order;

    public:
        // Constructor
        CoefficientBase(int order) : _order(order) {}

        // bool updateHessianMatrix(OsqpEigen::Solver& solver,
        //                  const Eigen::DiagonalMatrix<c_float, 1>& Q,
        //                  const Eigen::DiagonalMatrix<c_float, 1>& R,
        //                  int mpcWindow,
        //                  int k);

        // Method to represent the object as a string
        inline std::string toString() const {
            return "CoefficientBase with order: " + std::to_string(_order);
        }

        int order() const {
            return _order;
        }

        // t_coefficient method
        std::vector<double> t_coefficient(double t) const;

        // t_first_derivative_coefficient method
        std::vector<double> t_first_derivative_coefficient(double t) const;

        // t_second_derivative_coefficient method
        std::vector<double> t_second_derivative_coefficient(double t) const;

        // t_third_derivative_coefficient method
        std::vector<double> t_third_derivative_coefficient(double t) const;
};

}