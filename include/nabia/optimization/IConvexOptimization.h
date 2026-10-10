//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_ICONVEXOPTIMIZATION_H
#define NABIA_ICONVEXOPTIMIZATION_H
#include "IOptimizer.h"
#include <vector>
#include <eigen/Eigen>

struct BoxConstraints{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct LinearConstraint {

};

struct ConvexOptimizationParams{
    size_t dim;
    double tolerance;
    double max_iter;
};
class IConvexOptimization: public IOptimizer<double> {};

#endif // NABIA_ICONVEXOPTIMIZATION_H
