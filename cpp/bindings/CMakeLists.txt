# Python bindings: the C++ side of the nabia_python_bridge wheel (pybind11).
#
#   nabia_python_bridge._core          the whole library (stats, signal, forecast, optimization, risk,
#                                      strategy, regression)
#   nabia_python_bridge._rb_backtest   JSON-configured risk budgeting backtests (apps/rb_backtest),
#                                      used by nabia.portfolio in the Python project
#
# Wheel build (pyproject.toml of this C++ project, scikit-build-core): both modules and the package's
# .py files are installed into the wheel. Development build (CLion, preset python-bridge): the modules
# are written next to the .py files in bindings/nabia_python_bridge/, so the package can be imported
# with PYTHONPATH=cpp/bindings, e.g. to debug the C++ side from a Python script.

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# pybind11: from the Python environment (pip install pybind11), else fetched.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
find_package(pybind11 2.13 CONFIG QUIET HINTS "${pybind11_cmake_dir}")
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching it")
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0.1
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_core
    src/module.cpp
    src/bind_stats.cpp
    src/bind_signal.cpp
    src/bind_forecast.cpp
    src/bind_optimization.cpp
    src/bind_risk.cpp
    src/bind_strategy.cpp
    src/bind_regression.cpp)
target_link_libraries(_core PRIVATE nabia::nabia)
nabia_set_warnings(_core)

pybind11_add_module(_rb_backtest src/rb_backtest_module.cpp)
target_link_libraries(_rb_backtest PRIVATE rb_backtest_core)
nabia_set_warnings(_rb_backtest)

if(SKBUILD)
    install(TARGETS _core _rb_backtest LIBRARY DESTINATION nabia_python_bridge)
    install(DIRECTORY nabia_python_bridge/ DESTINATION nabia_python_bridge
        FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
else()
    set_target_properties(_core _rb_backtest PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nabia_python_bridge)
endif()
