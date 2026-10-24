// Helpers shared by the binding files of nabia_python_bridge._core.
#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nabia_bridge {

namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

/// Default-constructs T (value-initialized, so fields without a default are
/// zero) and assigns every keyword argument to the attribute of that name.
/// Unknown names raise AttributeError, so typos do not pass silently.
template <typename T>
std::unique_ptr<T> from_kwargs(const py::kwargs& kwargs) {
    auto value = std::make_unique<T>();
    {
        const py::object self = py::cast(value.get(), py::return_value_policy::reference);
        for (const auto& [name, item] : kwargs) {
            py::setattr(self, name, item);
        }
    }
    return value;
}

/// py::init for a parameters struct: T(**kwargs).
template <typename T>
auto kwargs_init() {
    return py::init([](const py::kwargs& kwargs) { return from_kwargs<T>(kwargs); });
}

/// A 1-d float64 array viewed as a span (the array must outlive the span).
inline std::span<const double> as_span(const DoubleArray& x) {
    if (x.ndim() != 1) {
        throw std::invalid_argument("expected a 1-dimensional array");
    }
    return {x.data(), static_cast<std::size_t>(x.size())};
}

inline DoubleArray to_array(const std::vector<double>& v) {
    DoubleArray out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

/// A Python callable that C++ code can copy and call from any thread while
/// the GIL is released: copies only share a pointer, and every call (and the
/// final release) takes the GIL.
class PyCallable {
public:
    explicit PyCallable(py::function f)
        : f_(new py::function(std::move(f)), [](py::function* p) {
              const py::gil_scoped_acquire gil;
              delete p;
          }) {}

    template <typename R, typename... Args>
    R call(const Args&... args) const {
        const py::gil_scoped_acquire gil;
        return (*f_)(args...).template cast<R>();
    }

private:
    std::shared_ptr<py::function> f_;
};

void bind_stats(py::module_& m);
void bind_signal(py::module_& m);
void bind_forecast(py::module_& m);
void bind_optimization(py::module_& m);
void bind_risk(py::module_& m);
void bind_strategy(py::module_& m);
void bind_regression(py::module_& m);

} // namespace nabia_bridge
