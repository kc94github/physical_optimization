/**
 * @file UpdateMatricesTest.cpp
 * @author Giulio Romualdi
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2020
 */

// Catch2
// #include <catch2/catch.hpp>

// OsqpEigen
#include <OsqpEigen/OsqpEigen.h>

// eigen
#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>

// colors
#define ANSI_TXT_GRN "\033[0;32m"
#define ANSI_TXT_MGT "\033[0;35m" // Magenta
#define ANSI_TXT_DFT "\033[0;0m"  // Console default
#define GTEST_BOX "[     cout ] "
#define COUT_GTEST ANSI_TXT_GRN << GTEST_BOX // You could add the Default
#define COUT_GTEST_MGT COUT_GTEST << ANSI_TXT_MGT

#define T 0.1

void setDynamicsMatrices(Eigen::Matrix<c_float, 2, 2> &a,
                         Eigen::Matrix<c_float, 2, 1> &b,
                         Eigen::Matrix<c_float, 1, 2> &c, double t) {

  double omega = 0.1132;
  double alpha = 0.5 * sin(2 * M_PI * omega * t);
  double beta = 2 - 1 * sin(2 * M_PI * omega * t);

  a << alpha, 1, 0, alpha;

  b << 0, 1;

  c << beta, 0;
}

void setWeightMatrices(Eigen::DiagonalMatrix<c_float, 1> &Q,
                       Eigen::DiagonalMatrix<c_float, 1> &R) {
  Q.diagonal() << 10;
  R.diagonal() << 1;
}

void castMPCToQPHessian(const Eigen::DiagonalMatrix<c_float, 1> &Q,
                        const Eigen::DiagonalMatrix<c_float, 1> &R,
                        int mpcWindow, int k,
                        Eigen::SparseMatrix<c_float> &hessianMatrix) {

  Eigen::Matrix<c_float, 2, 2> a;
  Eigen::Matrix<c_float, 2, 1> b;
  Eigen::Matrix<c_float, 1, 2> c;

  hessianMatrix.resize(2 * (mpcWindow + 1) + 1 * mpcWindow,
                       2 * (mpcWindow + 1) + 1 * mpcWindow);

  // populate hessian matrix
  for (int i = 0; i < 2 * (mpcWindow + 1) + 1 * mpcWindow; i++) {
    double t = (k + i) * T;
    setDynamicsMatrices(a, b, c, t);
    if (i < 2 * (mpcWindow + 1)) {
      // here the structure of the matrix c is used!
      int pos = i % 2;
      float value = c(pos) * Q.diagonal()[0] * c(pos);
      if (value != 0)
        hessianMatrix.insert(i, i) = value;
    } else {
      float value = R.diagonal()[0];
      if (value != 0)
        hessianMatrix.insert(i, i) = value;
    }
  }
}

void castMPCToQPGradient(const Eigen::DiagonalMatrix<c_float, 1> &Q,
                         const Eigen::Matrix<c_float, 1, 1> &yRef,
                         int mpcWindow, int k,
                         Eigen::Matrix<c_float, -1, 1> &gradient) {

  Eigen::Matrix<c_float, 2, 2> a;
  Eigen::Matrix<c_float, 2, 1> b;
  Eigen::Matrix<c_float, 1, 2> c;

  Eigen::Matrix<c_float, 1, 1> Qy_ref;
  Qy_ref = Q * (-yRef);

  // populate the gradient vector
  gradient = Eigen::Matrix<c_float, -1, 1>::Zero(
      2 * (mpcWindow + 1) + 1 * mpcWindow, 1);
  for (int i = 0; i < 2 * (mpcWindow + 1); i++) {
    double t = (k + i) * T;
    setDynamicsMatrices(a, b, c, t);

    int pos = i % 2;
    float value = Qy_ref(0, 0) * c(pos);
    gradient(i, 0) = value;
  }
}

void castMPCToQPConstraintMatrix(
    int mpcWindow, int k, Eigen::SparseMatrix<c_float> &constraintMatrix) {
  constraintMatrix.resize(2 * (mpcWindow + 1),
                          2 * (mpcWindow + 1) + 1 * mpcWindow);

  // populate linear constraint matrix
  for (int i = 0; i < 2 * (mpcWindow + 1); i++) {
    constraintMatrix.insert(i, i) = -1;
  }

  Eigen::Matrix<c_float, 2, 2> a;
  Eigen::Matrix<c_float, 2, 1> b;
  Eigen::Matrix<c_float, 1, 2> c;

  for (int i = 0; i < mpcWindow; i++) {
    double t = (k + i) * T;
    setDynamicsMatrices(a, b, c, t);
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        float value = a(j, k);
        if (value != 0) {
          constraintMatrix.insert(2 * (i + 1) + j, 2 * i + k) = value;
        }
      }
  }

  for (int i = 0; i < mpcWindow; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 1; k++) {
        // b is constant
        float value = b(j, k);
        if (value != 0) {
          constraintMatrix.insert(2 * (i + 1) + j,
                                  1 * i + k + 2 * (mpcWindow + 1)) = value;
        }
      }
}

void castMPCToQPConstraintVectors(const Eigen::Matrix<c_float, 2, 1> &x0,
                                  int mpcWindow,
                                  Eigen::Matrix<c_float, -1, 1> &lowerBound,
                                  Eigen::Matrix<c_float, -1, 1> &upperBound) {
  // evaluate the lower and the upper equality vectors
  lowerBound = Eigen::Matrix<c_float, -1, -1>::Zero(2 * (mpcWindow + 1), 1);
  lowerBound.block(0, 0, 2, 1) = -x0;
  upperBound = lowerBound;
}

bool updateHessianMatrix(OsqpEigen::Solver &solver,
                         const Eigen::DiagonalMatrix<c_float, 1> &Q,
                         const Eigen::DiagonalMatrix<c_float, 1> &R,
                         int mpcWindow, int k) {
  Eigen::SparseMatrix<c_float> hessianMatrix;
  castMPCToQPHessian(Q, R, mpcWindow, k, hessianMatrix);

  if (!solver.updateHessianMatrix(hessianMatrix))
    return false;

  return true;
}

bool updateLinearConstraintsMatrix(OsqpEigen::Solver &solver, int mpcWindow,
                                   int k) {
  Eigen::SparseMatrix<c_float> constraintMatrix;
  castMPCToQPConstraintMatrix(mpcWindow, k, constraintMatrix);

  if (!solver.updateLinearConstraintsMatrix(constraintMatrix))
    return false;

  return true;
}

void updateConstraintVectors(const Eigen::Matrix<c_float, 2, 1> &x0,
                             Eigen::Matrix<c_float, -1, 1> &lowerBound,
                             Eigen::Matrix<c_float, -1, 1> &upperBound) {
  lowerBound.block(0, 0, 2, 1) = -x0;
  upperBound.block(0, 0, 2, 1) = -x0;
}

int main() {
  // open the ofstream
  std::ofstream dataStream;
  dataStream.open("output.txt");

  // set the preview window
  int mpcWindow = 100;

  // allocate the dynamics matrices
  Eigen::Matrix<c_float, 2, 2> a;
  Eigen::Matrix<c_float, 2, 1> b;
  Eigen::Matrix<c_float, 1, 2> c;

  // allocate the weight matrices
  Eigen::DiagonalMatrix<c_float, 1> Q;
  Eigen::DiagonalMatrix<c_float, 1> R;

  // allocate the initial and the reference state space
  Eigen::Matrix<c_float, 2, 1> x0;
  Eigen::Matrix<c_float, 1, 1> yRef;
  Eigen::Matrix<c_float, 1, 1> y;

  // allocate QP problem matrices and vectores
  Eigen::SparseMatrix<c_float> hessian;
  Eigen::Matrix<c_float, -1, 1> gradient;
  Eigen::SparseMatrix<c_float> linearMatrix;
  Eigen::Matrix<c_float, -1, 1> lowerBound;
  Eigen::Matrix<c_float, -1, 1> upperBound;

  // set the initial and the desired states
  x0 << 0, 0;
  yRef << 1;

  // set MPC problem quantities
  setWeightMatrices(Q, R);

  // cast the MPC problem as QP problem
  castMPCToQPHessian(Q, R, mpcWindow, 0, hessian);
  castMPCToQPGradient(Q, yRef, mpcWindow, 0, gradient);
  castMPCToQPConstraintMatrix(mpcWindow, 0, linearMatrix);
  castMPCToQPConstraintVectors(x0, mpcWindow, lowerBound, upperBound);

  // instantiate the solver
  OsqpEigen::Solver solver;

  // settings
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);

  // set the initial data of the QP solver
  solver.data()->setNumberOfVariables(2 * (mpcWindow + 1) + 1 * mpcWindow);
  solver.data()->setNumberOfConstraints(2 * (mpcWindow + 1));
  solver.data()->setHessianMatrix(hessian);
  solver.data()->setGradient(gradient);
  solver.data()->setLinearConstraintsMatrix(linearMatrix);
  solver.data()->setLowerBound(lowerBound);
  solver.data()->setUpperBound(upperBound);

  // instantiate the solver
  solver.initSolver();

  // controller input and QPSolution vector
  Eigen::Matrix<c_float, -1, 1> ctr;
  Eigen::Matrix<c_float, -1, 1> QPSolution;

  // number of iteration steps
  int numberOfSteps = 50;

  // profiling quantities
  clock_t startTime, endTime;
  double avarageTime = 0;

  for (int i = 0; i < numberOfSteps; i++) {
    startTime = clock();

    setDynamicsMatrices(a, b, c, i * T);

    // update the constraint bound
    updateHessianMatrix(solver, Q, R, mpcWindow, i);
    updateLinearConstraintsMatrix(solver, mpcWindow, i);

    castMPCToQPGradient(Q, yRef, mpcWindow, i, gradient);
    solver.updateGradient(gradient);

    updateConstraintVectors(x0, lowerBound, upperBound);
    solver.updateBounds(lowerBound, upperBound);

    // solve the QP problem
    solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError;

    // get the controller input
    QPSolution = solver.getSolution();
    ctr = QPSolution.block(2 * (mpcWindow + 1), 0, 1, 1);

    // save data into file
    auto x0Data = x0.data();
    for (int j = 0; j < 2; j++)
      dataStream << x0Data[j] << " ";
    dataStream << std::endl;

    // propagate the model
    x0 = a * x0 + b * ctr;
    y = c * x0;

    endTime = clock();

    avarageTime += static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
  }

  // close the stream
  dataStream.close();

  std::cout << COUT_GTEST_MGT
            << "Avarage time = " << avarageTime / numberOfSteps << " seconds."
            << ANSI_TXT_DFT << std::endl;
}