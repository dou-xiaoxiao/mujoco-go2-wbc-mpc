#include "go2wbc/OsqpSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace go2wbc {

namespace {

const double kSparseTolerance = 1.0e-12;

std::vector<OSQPFloat> copyEigenVector(const VectorX& vector) {
    std::vector<OSQPFloat> out(static_cast<size_t>(vector.size()));
    for (int i = 0; i < vector.size(); ++i) {
        double value = vector(i);
        if (std::isinf(value) && value < 0.0) {
            value = -OSQP_INFTY;
        } else if (std::isinf(value) && value > 0.0) {
            value = OSQP_INFTY;
        }
        out[static_cast<size_t>(i)] = static_cast<OSQPFloat>(value);
    }
    return out;
}

}  // namespace

SparseCSC denseToCSC(const MatrixX& dense, bool upper_triangle_only) {
    SparseCSC out;
    out.rows = static_cast<int>(dense.rows());
    out.cols = static_cast<int>(dense.cols());
    out.col_ptr.resize(static_cast<size_t>(out.cols + 1), 0);

    for (int col = 0; col < out.cols; ++col) {
        out.col_ptr[static_cast<size_t>(col)] = static_cast<OSQPInt>(out.values.size());
        int row_start = upper_triangle_only ? 0 : 0;
        int row_end = upper_triangle_only ? std::min(col, out.rows - 1) : out.rows - 1;
        for (int row = row_start; row <= row_end; ++row) {
            double value = dense(row, col);
            if (std::abs(value) > kSparseTolerance) {
                out.row_idx.push_back(static_cast<OSQPInt>(row));
                out.values.push_back(static_cast<OSQPFloat>(value));
            }
        }
    }
    out.col_ptr[static_cast<size_t>(out.cols)] = static_cast<OSQPInt>(out.values.size());
    return out;
}

OsqpSolver::OsqpSolver()
    : solver_(0),
      n_(0),
      m_(0),
      p_nnz_(0),
      a_nnz_(0),
      eps_abs_(1.0e-6),
      eps_rel_(1.0e-6),
      max_iter_(10000),
      polishing_(true) {}

OsqpSolver::~OsqpSolver() {
    cleanup();
}

void OsqpSolver::setTolerances(double eps_abs, double eps_rel) {
    eps_abs_ = eps_abs;
    eps_rel_ = eps_rel;
}

void OsqpSolver::setMaxIterations(int max_iter) {
    max_iter_ = max_iter;
}

void OsqpSolver::setPolishing(bool enabled) {
    polishing_ = enabled;
}

QpSolution OsqpSolver::solve(const QpProblem& problem) {
    if (problem.P.rows != problem.P.cols) {
        throw std::runtime_error("OSQP requires a square P matrix.");
    }
    if (problem.A.cols != problem.P.cols) {
        throw std::runtime_error("A and P have inconsistent variable dimensions.");
    }
    if (problem.q.size() != problem.P.cols ||
        problem.lower.size() != problem.A.rows ||
        problem.upper.size() != problem.A.rows) {
        throw std::runtime_error("QP vector dimensions are inconsistent.");
    }

    if (solver_ == 0 || !sameStructure(problem)) {
        setup(problem);
    } else {
        update(problem);
    }

    if (last_x_.size() == problem.q.size()) {
        osqp_warm_start(solver_, last_x_.data(), OSQP_NULL);
    }

    OSQPInt flag = osqp_solve(solver_);
    if (flag != 0) {
        throw std::runtime_error("osqp_solve failed with flag " + std::to_string(static_cast<int>(flag)));
    }

    QpSolution solution;
    solution.x = VectorX::Zero(problem.q.size());
    if (solver_->solution != 0 && solver_->solution->x != 0) {
        for (int i = 0; i < solution.x.size(); ++i) {
            solution.x(i) = solver_->solution->x[i];
        }
        last_x_ = solution.x;
    }
    if (solver_->info != 0) {
        solution.status = solver_->info->status;
        solution.status_value = static_cast<int>(solver_->info->status_val);
        solution.objective = solver_->info->obj_val;
        solution.iterations = static_cast<int>(solver_->info->iter);
    }
    return solution;
}

bool OsqpSolver::sameStructure(const QpProblem& problem) const {
    return problem.P.cols == n_
        && problem.A.rows == m_
        && static_cast<int>(problem.P.values.size()) == p_nnz_
        && static_cast<int>(problem.A.values.size()) == a_nnz_
        && problem.P.col_ptr == p_data_.col_ptr
        && problem.P.row_idx == p_data_.row_idx
        && problem.A.col_ptr == a_data_.col_ptr
        && problem.A.row_idx == a_data_.row_idx;
}

void OsqpSolver::setup(const QpProblem& problem) {
    cleanup();
    storeProblemData(problem);

    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.verbose = 0;
    settings.warm_starting = 1;
    settings.polishing = polishing_ ? 1 : 0;
    settings.eps_abs = eps_abs_;
    settings.eps_rel = eps_rel_;
    settings.max_iter = max_iter_;

    OSQPCscMatrix_set_data(
        &p_csc_,
        p_data_.rows,
        p_data_.cols,
        static_cast<OSQPInt>(p_data_.values.size()),
        p_data_.values.data(),
        p_data_.row_idx.data(),
        p_data_.col_ptr.data()
    );
    OSQPCscMatrix_set_data(
        &a_csc_,
        a_data_.rows,
        a_data_.cols,
        static_cast<OSQPInt>(a_data_.values.size()),
        a_data_.values.data(),
        a_data_.row_idx.data(),
        a_data_.col_ptr.data()
    );

    OSQPInt flag = osqp_setup(
        &solver_,
        &p_csc_,
        q_data_.data(),
        &a_csc_,
        l_data_.data(),
        u_data_.data(),
        static_cast<OSQPInt>(m_),
        static_cast<OSQPInt>(n_),
        &settings
    );
    if (flag != 0) {
        cleanup();
        throw std::runtime_error("osqp_setup failed with flag " + std::to_string(static_cast<int>(flag)));
    }
}

void OsqpSolver::update(const QpProblem& problem) {
    storeProblemData(problem);
    OSQPInt vec_flag = osqp_update_data_vec(solver_, q_data_.data(), l_data_.data(), u_data_.data());
    if (vec_flag != 0) {
        setup(problem);
        return;
    }
    OSQPInt mat_flag = osqp_update_data_mat(
        solver_,
        p_data_.values.data(),
        OSQP_NULL,
        static_cast<OSQPInt>(p_data_.values.size()),
        a_data_.values.data(),
        OSQP_NULL,
        static_cast<OSQPInt>(a_data_.values.size())
    );
    if (mat_flag != 0) {
        setup(problem);
    }
}

void OsqpSolver::cleanup() {
    if (solver_ != 0) {
        osqp_cleanup(solver_);
        solver_ = 0;
    }
}

void OsqpSolver::storeProblemData(const QpProblem& problem) {
    p_data_ = problem.P;
    a_data_ = problem.A;
    q_data_ = copyEigenVector(problem.q);
    l_data_ = copyEigenVector(problem.lower);
    u_data_ = copyEigenVector(problem.upper);
    n_ = problem.P.cols;
    m_ = problem.A.rows;
    p_nnz_ = static_cast<int>(problem.P.values.size());
    a_nnz_ = static_cast<int>(problem.A.values.size());
}

}  // namespace go2wbc
