//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/signal/ButterworthBandPass.h"

#include <Iir.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

using Filter = Iir::Butterworth::BandPass<ButterworthBandPass::max_order>;

/// Designs `filter` in place. iir1 filters hold a pointer to their own
/// coefficient storage, so they must not be copied: always design a new one.
void design(Filter& filter, const ButterworthBandPassParams& p) {
    // iir1 takes the band as centre and width; its edges are centre -+ width / 2.
    filter.setup(p.order, p.sample_rate, 0.5 * (p.low_cutoff + p.high_cutoff), p.high_cutoff - p.low_cutoff);
}

} // namespace

struct ButterworthBandPass::Impl {
    Filter filter;
};

ButterworthBandPass::ButterworthBandPass(std::unique_ptr<ButterworthBandPassParams> parameters)
    : IFilter(std::move(parameters)) {
    if (!parameters_) {
        throw std::invalid_argument("ButterworthBandPass: parameters are null");
    }
    const ButterworthBandPassParams& p = *parameters_;
    if (p.order < 1 || p.order > max_order) {
        throw std::invalid_argument("ButterworthBandPass: order must be in [1, max_order]");
    }
    if (!(p.sample_rate > 0.0 && std::isfinite(p.sample_rate))) {
        throw std::invalid_argument("ButterworthBandPass: sample rate must be positive");
    }
    if (!(p.low_cutoff > 0.0 && p.low_cutoff < p.high_cutoff && p.high_cutoff < 0.5 * p.sample_rate)) {
        throw std::invalid_argument("ButterworthBandPass: need 0 < low_cutoff < high_cutoff < sample_rate / 2");
    }
    impl_ = std::make_unique<Impl>();
    design(impl_->filter, p);
}

ButterworthBandPass::~ButterworthBandPass() = default;
ButterworthBandPass::ButterworthBandPass(ButterworthBandPass&&) noexcept = default;
ButterworthBandPass& ButterworthBandPass::operator=(ButterworthBandPass&&) noexcept = default;

double ButterworthBandPass::process(double sample) {
    return impl_->filter.filter(sample);
}

void ButterworthBandPass::reset() {
    impl_->filter.reset();
}

void ButterworthBandPass::apply(Eigen::VectorXd& signal) const {
    const auto fresh = std::make_unique<Filter>();  // large (fixed max order): keep it off the stack
    design(*fresh, *parameters_);
    for (Eigen::Index t = 0; t < signal.size(); ++t) {
        signal(t) = fresh->filter(signal(t));
    }
}
