#include <vector>
#include "coefficient_base.h"

namespace Geometry {

class Polynomial1d : public CoefficientBase {
    private:
        int _order;
        int _param_size;
        std::vector<double> _coefficients;

    public:
        Polynomial1d(const std::vector<double>& coefficients):CoefficientBase(coefficients.size()-1), _coefficients(coefficients),_order(coefficients.size()-1),_param_size(coefficients.size()) {}

        static Polynomial1d polynomial_from_coefficients(const std::vector<double>& coefficients) {
            return Polynomial1d(coefficients);
        }

        inline std::string toString() const {
            return "Polynomial1d with order: " + std::to_string(_order);
        }

        inline int order() const {
            return _order;
        }

        inline int param_size() const {
            return _param_size;
        }

        inline std::vector<double> coefficients() const {
            return _coefficients;
        }

        bool operator==(const Polynomial1d& other) const {
            return _coefficients == other._coefficients; // Compare the value of both instances
        }

        template<typename CoefficientFunc, typename... Args>
        double summation(CoefficientFunc coefficient_func, Args... args) {
            double total = 0.0;
            std::vector<double> res = coefficient_func(args...);
            for (int i=0;i<res.size();i++) {
                total += _coefficients[i] * res[i];
            }
            return total;
        }

        double evaluate(const double& t);

        double derivative(const double& t);

        double second_derivative(const double& t);

        double third_derivative(const double& t);

        Polynomial1d derivative_polynomial(const uint& order);

        Polynomial1d integral_polynomial(const uint& order);

};

}