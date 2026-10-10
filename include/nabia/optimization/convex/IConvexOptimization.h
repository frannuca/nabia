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

class IConvexOptimization: public IOptimizer<ConvexOptimizationParams> {
public:
    using IOptimizer::IOptimizer;
};

#endif // NABIA_ICONVEXOPTIMIZATION_H
