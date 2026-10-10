//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_EWMA_H
#define NABIA_EWMA_H
#include "nabia/signal/IFilter.h"
#include <memory>
#include <Eigen/Dense>

struct EWMAParams {
    double alpha = 0.1;                  // smoothing factor in (0, 1]; 1 = no smoothing
    bool seed_with_first_sample = true;  // y[0] = x[0]; otherwise start from y[-1] = 0, so y[0] = alpha x[0]
};

/// Exponentially weighted moving average, a causal one-pole low-pass:
///     y[t] = alpha x[t] + (1 - alpha) y[t-1],   H(z) = alpha / (1 - (1 - alpha) z^-1).
/// Implemented with iir1 (Iir::Custom::OnePole, direct form I).
///
/// With seed_with_first_sample (the default), the filter starts at the first
/// input instead of 0, which avoids the start-up bias towards zero. The result
/// matches nabia::signal::exponential_moving_average.
class EWMA: public IFilter<EWMAParams> {
public:
    /// Throws std::invalid_argument unless 0 < alpha <= 1.
    explicit EWMA(std::unique_ptr<EWMAParams> parameters);
    ~EWMA() override;
    EWMA(EWMA&&) noexcept;
    EWMA& operator=(EWMA&&) noexcept;

    double process(double sample) override;
    void reset() override;
    void apply(Eigen::VectorXd& signal) const override;

    /// alpha giving the same centre of mass as a simple moving average of `span` samples: 2 / (span + 1).
    [[nodiscard]] static double alpha_from_span(double span);
    /// alpha for which a sample's weight halves after `halflife` samples: 1 - 2^(-1 / halflife).
    [[nodiscard]] static double alpha_from_halflife(double halflife);

private:
    struct Impl;  // holds the iir1 filter; keeps iir1 out of this header
    std::unique_ptr<Impl> impl_;
};

#endif // NABIA_EWMA_H
