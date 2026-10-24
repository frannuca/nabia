//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_BUTTERWORTHBANDPASS_H
#define NABIA_BUTTERWORTHBANDPASS_H
#include "nabia/signal/IFilter.h"
#include <memory>
#include <Eigen/Dense>

struct ButterworthBandPassParams {
    int order = 4;             // prototype order N (the band-pass has 2N poles), 1..max_order
    double sample_rate = 1.0;  // fs, in samples per unit of time (e.g. Hz)
    double low_cutoff = 0.0;   // lower -3 dB edge, same unit as fs, 0 < low < high
    double high_cutoff = 0.0;  // upper -3 dB edge, high < fs / 2 (Nyquist)
};

/// Causal Butterworth band-pass filter, a wrapper over iir1's
/// Iir::Butterworth::BandPass (bilinear transform, cascade of biquads in
/// direct form II).
///
/// The magnitude response is maximally flat in the pass band, with gain 1 at
/// the centre and -3 dB (gain 1/sqrt(2)) at the cut-offs. Being causal (not
/// zero-phase), it delays the signal, most of all near the band edges; the
/// output starts from a zero state, so expect a transient of a few cycles of
/// the lowest pass-band frequency.
class ButterworthBandPass: public IFilter<ButterworthBandPassParams> {
public:
    static constexpr int max_order = 16;

    /// Throws std::invalid_argument unless 1 <= order <= max_order, fs > 0 and
    /// 0 < low_cutoff < high_cutoff < fs / 2.
    explicit ButterworthBandPass(std::unique_ptr<ButterworthBandPassParams> parameters);
    ~ButterworthBandPass() override;
    ButterworthBandPass(ButterworthBandPass&&) noexcept;
    ButterworthBandPass& operator=(ButterworthBandPass&&) noexcept;

    double process(double sample) override;
    void reset() override;
    void apply(Eigen::VectorXd& signal) const override;

private:
    struct Impl;  // holds the iir1 filter; keeps iir1 out of this header
    std::unique_ptr<Impl> impl_;
};

#endif // NABIA_BUTTERWORTHBANDPASS_H
