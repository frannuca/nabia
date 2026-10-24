//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/signal/EWMA.h"

#include <Iir.h>

#include <cmath>
#include <stdexcept>
#include <utility>

/// One-pole section plus its state. With seeding, the filter runs on
/// x - x[0] and adds x[0] back: since the DC gain is 1, this equals starting
/// from y[-1] = x[0], so y[0] = x[0].
struct EWMA::Impl {
    Iir::Custom::OnePole section;
    Iir::DirectFormI state;
    bool seed = true;
    bool started = false;
    double offset = 0.0;

    double process(double x) {
        if (!started) {
            offset = seed ? x : 0.0;
            started = true;
        }
        return section.filter(x - offset, state) + offset;
    }

    void reset() {
        state.reset();
        started = false;
        offset = 0.0;
    }
};

EWMA::EWMA(std::unique_ptr<EWMAParams> parameters) : IFilter(std::move(parameters)) {
    if (!parameters_) {
        throw std::invalid_argument("EWMA: parameters are null");
    }
    const double alpha = parameters_->alpha;
    if (!(alpha > 0.0 && alpha <= 1.0)) {
        throw std::invalid_argument("EWMA: alpha must be in (0, 1]");
    }
    impl_ = std::make_unique<Impl>();
    // H(z) = alpha (1 - 0 z^-1) / (1 - (1 - alpha) z^-1)
    impl_->section.setup(alpha, 1.0 - alpha, 0.0);
    impl_->seed = parameters_->seed_with_first_sample;
}

EWMA::~EWMA() = default;
EWMA::EWMA(EWMA&&) noexcept = default;
EWMA& EWMA::operator=(EWMA&&) noexcept = default;

double EWMA::process(double sample) {
    return impl_->process(sample);
}

void EWMA::reset() {
    impl_->reset();
}

void EWMA::apply(Eigen::VectorXd& signal) const {
    Impl fresh = *impl_;  // OnePole holds plain coefficients, so copying is safe
    fresh.reset();
    for (Eigen::Index t = 0; t < signal.size(); ++t) {
        signal(t) = fresh.process(signal(t));
    }
}

double EWMA::alpha_from_span(double span) {
    if (!(span >= 1.0)) {
        throw std::invalid_argument("EWMA: span must be >= 1");
    }
    return 2.0 / (span + 1.0);
}

double EWMA::alpha_from_halflife(double halflife) {
    if (!(halflife > 0.0)) {
        throw std::invalid_argument("EWMA: halflife must be positive");
    }
    return 1.0 - std::exp2(-1.0 / halflife);
}
