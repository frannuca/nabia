#include <nabia/forecast/exponential_smoothing.hpp>
#include <nabia/signal/smoothing.hpp>
#include <nabia/stats/descriptive.hpp>

#include <iostream>
#include <vector>

int main() {
    // A small noisy, roughly level series.
    const std::vector<double> series{
        102.0, 98.5, 101.2, 103.8, 99.1, 100.4, 104.2, 97.9, 101.7, 100.9,
    };

    std::cout << "mean   = " << nabia::stats::mean(series) << '\n'
              << "stddev = " << nabia::stats::stddev(series) << '\n'
              << "acf(1) = " << nabia::stats::autocorrelation(series, 1) << "\n\n";

    std::cout << "3-point moving average:";
    for (double v : nabia::signal::moving_average(series, 3)) {
        std::cout << ' ' << v;
    }
    std::cout << "\n\n";

    const Eigen::Map<const Eigen::VectorXd> y(series.data(), static_cast<Eigen::Index>(series.size()));

    // Pick the alpha with the lowest in-sample error.
    double best_alpha = 0.0;
    double best_sse = 0.0;
    for (double alpha : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        nabia::forecast::SimpleExponentialSmoothing model(alpha);
        model.fit(y);
        std::cout << "alpha = " << alpha << "  SSE = " << model.sse() << '\n';
        if (best_alpha == 0.0 || model.sse() < best_sse) {
            best_alpha = alpha;
            best_sse = model.sse();
        }
    }

    nabia::forecast::SimpleExponentialSmoothing best(best_alpha);
    best.fit(y);
    const Forecast fc = best.forecast(3);
    const Eigen::VectorXd lo = fc.lower(0.95);
    const Eigen::VectorXd hi = fc.upper(0.95);

    std::cout << "\nbest alpha = " << best_alpha << ", forecast (95% interval):\n";
    for (Eigen::Index h = 0; h < fc.horizon(); ++h) {
        std::cout << "  t+" << h + 1 << " = " << fc.mean(h) << "  [" << lo(h) << ", " << hi(h) << "]\n";
    }
    std::cout << '\n';
}
