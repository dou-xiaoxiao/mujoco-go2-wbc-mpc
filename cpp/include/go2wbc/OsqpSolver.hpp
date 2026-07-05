#pragma once

#include <string>
#include <vector>

#include <osqp.h>

#include "go2wbc/Types.hpp"

namespace go2wbc {

struct SparseCSC {
    int rows;
    int cols;
    std::vector<OSQPInt> col_ptr;
    std::vector<OSQPInt> row_idx;
    std::vector<OSQPFloat> values;

    SparseCSC() : rows(0), cols(0) {}
};

struct QpProblem {
    SparseCSC P;
    VectorX q;
    SparseCSC A;
    VectorX lower;
    VectorX upper;
};

struct QpSolution {
    VectorX x;
    std::string status;
    int status_value;
    double objective;
    int iterations;

    QpSolution() : status_value(0), objective(0.0), iterations(0) {}
};

class OsqpSolver {
public:
    OsqpSolver();
    ~OsqpSolver();

    OsqpSolver(const OsqpSolver&) = delete;
    OsqpSolver& operator=(const OsqpSolver&) = delete;

    void setTolerances(double eps_abs, double eps_rel);
    void setMaxIterations(int max_iter);
    void setPolishing(bool enabled);
    QpSolution solve(const QpProblem& problem);

private:
    bool sameStructure(const QpProblem& problem) const;
    void setup(const QpProblem& problem);
    void update(const QpProblem& problem);
    void cleanup();
    void storeProblemData(const QpProblem& problem);

    OSQPSolver* solver_;
    OSQPCscMatrix p_csc_;
    OSQPCscMatrix a_csc_;

    SparseCSC p_data_;
    SparseCSC a_data_;
    std::vector<OSQPFloat> q_data_;
    std::vector<OSQPFloat> l_data_;
    std::vector<OSQPFloat> u_data_;
    VectorX last_x_;

    int n_;
    int m_;
    int p_nnz_;
    int a_nnz_;
    double eps_abs_;
    double eps_rel_;
    int max_iter_;
    bool polishing_;
};

SparseCSC denseToCSC(const MatrixX& dense, bool upper_triangle_only);

}  // namespace go2wbc
