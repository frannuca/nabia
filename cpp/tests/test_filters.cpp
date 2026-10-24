#include <nabia/signal/ButterworthBandPass.h>
#include <nabia/signal/EWMA.h>
#include <nabia/signal/smoothing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Eigen::VectorXd;

namespace {

constexpr double kFs = 1000.0;  // Hz

std::unique_ptr<ButterworthBandPassParams> band(int order, double low, double high) {
    auto p = std::make_unique<ButterworthBandPassParams>();
    p->order = order;
    p->sample_rate = kFs;
    p->low_cutoff = low;
    p->high_cutoff = high;
    return p;
}

std::unique_ptr<EWMAParams> ewma(double alpha, bool seed = true) {
    auto p = std::make_unique<EWMAParams>();
    p->alpha = alpha;
    p->seed_with_first_sample = seed;
    return p;
}

VectorXd sine(double freq, Eigen::Index n) {
    VectorXd x(n);
    for (Eigen::Index t = 0; t < n; ++t) {
        x(t) = std::sin(2.0 * std::numbers::pi * freq * static_cast<double>(t) / kFs);
    }
    return x;
}

// Steady-state gain: peak output amplitude over the last second, after the transient.
double gain(const ButterworthBandPass& filter, double freq) {
    VectorXd x = sine(freq, 10000);
    filter.apply(x);
    return x.tail(1000).cwiseAbs().maxCoeff();
}

} // namespace

TEST_CASE("butterworth band-pass: frequency response", "[signal]") {
    const ButterworthBandPass bp(band(4, 40.0, 60.0));
    const double centre = std::sqrt(40.0 * 60.0);  // peak of the response (geometric centre after prewarping)
    CHECK_THAT(gain(bp, centre), WithinAbs(1.0, 0.01));
    CHECK_THAT(gain(bp, 40.0), WithinAbs(1.0 / std::numbers::sqrt2, 0.02));  // -3 dB edges
    CHECK_THAT(gain(bp, 60.0), WithinAbs(1.0 / std::numbers::sqrt2, 0.02));
    CHECK(gain(bp, 10.0) < 1e-4);   // stop band, below and above
    CHECK(gain(bp, 200.0) < 1e-4);

    SECTION("higher order gives a sharper roll-off") {
        const ButterworthBandPass low_order(band(2, 40.0, 60.0));
        CHECK(gain(bp, 30.0) < gain(low_order, 30.0));
    }
}

TEST_CASE("butterworth band-pass: removes DC and is causal", "[signal]") {
    const ButterworthBandPass bp(band(3, 5.0, 50.0));
    VectorXd dc = VectorXd::Constant(20000, 3.0);
    bp.apply(dc);
    CHECK(std::abs(dc(dc.size() - 1)) < 1e-6);

    // Impulse at t = 100: the output is exactly zero before it.
    VectorXd impulse = VectorXd::Zero(400);
    impulse(100) = 1.0;
    bp.apply(impulse);
    CHECK(impulse.head(100).isZero(0.0));
    CHECK(impulse(100) != 0.0);
}

TEST_CASE("butterworth band-pass: streaming matches batch", "[signal]") {
    ButterworthBandPass bp(band(4, 20.0, 80.0));
    VectorXd x = sine(30.0, 500) + 0.5 * sine(150.0, 500);
    VectorXd batch = x;
    bp.apply(batch);

    VectorXd streamed(x.size());
    for (Eigen::Index t = 0; t < x.size(); ++t) {
        streamed(t) = bp.process(x(t));
    }
    CHECK(streamed == batch);

    // apply() does not disturb the streaming state ...
    const double next = bp.process(1.0);
    bp.reset();
    VectorXd again = x;
    bp.apply(again);
    CHECK(again == batch);
    // ... and reset() returns to the initial state.
    CHECK(bp.process(x(0)) == batch(0));
    CHECK(next != batch(0));

    // Moving the filter keeps its state.
    ButterworthBandPass moved = std::move(bp);
    CHECK(moved.process(x(1)) == batch(1));
    CHECK(moved.parameters().order == 4);
}

TEST_CASE("butterworth band-pass: invalid parameters", "[signal]") {
    CHECK_THROWS_AS(ButterworthBandPass(band(0, 10.0, 20.0)), std::invalid_argument);
    CHECK_THROWS_AS(ButterworthBandPass(band(ButterworthBandPass::max_order + 1, 10.0, 20.0)), std::invalid_argument);
    CHECK_THROWS_AS(ButterworthBandPass(band(4, 20.0, 10.0)), std::invalid_argument);
    CHECK_THROWS_AS(ButterworthBandPass(band(4, 0.0, 10.0)), std::invalid_argument);
    CHECK_THROWS_AS(ButterworthBandPass(band(4, 100.0, 500.0)), std::invalid_argument);  // at Nyquist
    CHECK_THROWS_AS(ButterworthBandPass(nullptr), std::invalid_argument);
}

TEST_CASE("ewma: matches the reference implementation", "[signal]") {
    const std::vector<double> data{10.0, 20.0, 30.0, 25.0, 5.0, 40.0, 41.0};
    for (double alpha : {0.1, 0.5, 0.9, 1.0}) {
        const std::vector<double> expected = nabia::signal::exponential_moving_average(data, alpha);
        VectorXd x = Eigen::Map<const VectorXd>(data.data(), static_cast<Eigen::Index>(data.size()));
        EWMA(ewma(alpha)).apply(x);
        for (Eigen::Index t = 0; t < x.size(); ++t) {
            CHECK_THAT(x(t), WithinAbs(expected[static_cast<std::size_t>(t)], 1e-12));
        }
    }
}

TEST_CASE("ewma: zero initial state and step response", "[signal]") {
    // Without seeding: y[t] = 1 - (1 - alpha)^(t + 1) for a unit step.
    const double alpha = 0.2;
    VectorXd step = VectorXd::Ones(30);
    EWMA(ewma(alpha, false)).apply(step);
    for (Eigen::Index t = 0; t < step.size(); ++t) {
        CHECK_THAT(step(t), WithinAbs(1.0 - std::pow(1.0 - alpha, static_cast<double>(t + 1)), 1e-14));
    }
    // With seeding a constant signal passes unchanged.
    VectorXd flat = VectorXd::Constant(30, 7.5);
    EWMA(ewma(alpha)).apply(flat);
    CHECK(flat.isApproxToConstant(7.5, 1e-14));
}

TEST_CASE("ewma: streaming, reset and helpers", "[signal]") {
    EWMA f(ewma(0.3));
    CHECK(f.process(4.0) == 4.0);  // seeded with the first sample
    CHECK_THAT(f.process(14.0), WithinAbs(0.3 * 14.0 + 0.7 * 4.0, 1e-14));
    f.reset();
    CHECK(f.process(-2.0) == -2.0);  // re-seeded after reset

    CHECK_THAT(EWMA::alpha_from_span(9.0), WithinAbs(0.2, 1e-15));
    // After `halflife` samples the weight of an old sample has halved.
    const double a = EWMA::alpha_from_halflife(10.0);
    CHECK_THAT(std::pow(1.0 - a, 10.0), WithinRel(0.5, 1e-12));

    CHECK_THROWS_AS(EWMA(ewma(0.0)), std::invalid_argument);
    CHECK_THROWS_AS(EWMA(ewma(1.5)), std::invalid_argument);
    CHECK_THROWS_AS(EWMA::alpha_from_span(0.5), std::invalid_argument);
}
