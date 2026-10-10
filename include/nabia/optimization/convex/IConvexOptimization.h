//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_ICONVEXOPTIMIZATION_H
#define NABIA_ICONVEXOPTIMIZATION_H
#include "nabia/optimization/IOptimizer.h"
#include "nabia/optimization/convex/Constraints.h"
#include <optional>
#include <Eigen/Dense>

struct ConvexOptimizationParams{
    Eigen::Index dim;
    double tolerance;
    double max_iter;
    std::optional<Constraints> constraints;  // std::nullopt = unconstrained
};

/// Interface of the constrained solvers; R is the solver's result type.
template<typename R>
class IConvexOptimization: public IOptimizer<ConvexOptimizationParams, R> {
public:
    using IOptimizer<ConvexOptimizationParams, R>::IOptimizer;
};

#endif // NABIA_ICONVEXOPTIMIZATION_H
