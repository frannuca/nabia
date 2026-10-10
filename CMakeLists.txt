cmake_minimum_required(VERSION 3.25)

# CMake >= 3.29: link each library once even when it is reached through several
# targets (avoids macOS "ignoring duplicate libraries" linker warnings).
if(POLICY CMP0156)
    cmake_policy(SET CMP0156 NEW)
endif()

project(nabia
    VERSION 0.1.0
    DESCRIPTION "Numerical library for systematic trend strategies"
    LANGUAGES CXX)

# --- Options -----------------------------------------------------------------
# Only build tests/examples by default when nabia is the top-level project
# (not when someone pulls it in via add_subdirectory / FetchContent).
option(NABIA_BUILD_TESTS    "Build unit tests"  ${PROJECT_IS_TOP_LEVEL})
option(NABIA_BUILD_EXAMPLES "Build examples"    ${PROJECT_IS_TOP_LEVEL})
option(NABIA_BUILD_APPS     "Build command-line applications" ${PROJECT_IS_TOP_LEVEL})
option(NABIA_BUILD_DOCS     "Build PDF documentation (needs pdflatex)" ${PROJECT_IS_TOP_LEVEL})
option(NABIA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# Export compile_commands.json for editors / clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ------------------------------------------------------------
# Eigen (header-only linear algebra). Prefer an installed copy (Homebrew,
# vcpkg, apt...); otherwise download it at configure time.
find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found, fetching it")
    include(FetchContent)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        5.0.1
        GIT_SHALLOW    TRUE
        SYSTEM)  # treat Eigen headers as system headers -> no warnings from them
    FetchContent_MakeAvailable(Eigen3)
endif()

# Threads: DifferentialEvolution evaluates its population in parallel.
find_package(Threads REQUIRED)

# iir1 (IIR filter design, https://github.com/berndporr/iir1): an installed copy
# if there is one, otherwise fetched and built as a static library. It is a
# PRIVATE dependency: the nabia filter classes hide it behind a pimpl.
find_package(iir CONFIG QUIET)
if(TARGET iir::iir_static)
    set(NABIA_IIR_TARGET iir::iir_static)
elseif(TARGET iir::iir)
    set(NABIA_IIR_TARGET iir::iir)
else()
    message(STATUS "iir1 not found, fetching it")
    include(FetchContent)
    set(IIR1_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(IIR1_BUILD_DEMO    OFF CACHE BOOL "" FORCE)
    set(iir_exclude "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        set(iir_exclude EXCLUDE_FROM_ALL)  # build only what nabia links (not the shared lib)
    endif()
    FetchContent_Declare(iir
        GIT_REPOSITORY https://github.com/berndporr/iir1.git
        GIT_TAG        1.9.5
        GIT_SHALLOW    TRUE
        SYSTEM         # treat iir1 headers as system headers -> no warnings from them
        ${iir_exclude})
    FetchContent_MakeAvailable(iir)
    set(NABIA_IIR_TARGET iir::iir_static)
endif()

# --- Library -----------------------------------------------------------------
add_library(nabia
    src/stats/descriptive.cpp
    src/stats/normal.cpp
    src/signal/smoothing.cpp
    src/signal/ButterworthBandPass.cpp
    src/signal/EWMA.cpp
    src/forecast/exponential_smoothing.cpp
    src/forecast/IForecaster.cpp
    src/forecast/KalmanFilter.cpp
    src/forecast/Kernels.cpp
    src/forecast/GaussianProcess.cpp
    src/strategy/PerformanceMetrics.cpp
    src/strategy/RiskBudgetingPortfolio.cpp
    src/optimization/convex/Constraints.cpp
    src/optimization/convex/AugmentedLagrangian.cpp
    src/optimization/convex/SQP.cpp
    src/optimization/nongradopt/NelderMead.cpp
    src/optimization/nongradopt/DifferentialEvolution.cpp
        include/nabia/optimization/IOptimizer.h
        include/nabia/optimization/convex/Constraints.h
        include/nabia/optimization/convex/IConvexOptimization.h
        include/nabia/optimization/convex/AugmentedLagrangian.h
        include/nabia/optimization/convex/SQP.h
        include/nabia/optimization/nongradopt/NelderMead.h
        include/nabia/optimization/nongradopt/DifferentialEvolution.h
        src/optimization/convex/LBFGS.cpp
        include/nabia/optimization/convex/LBFGS.h
        include/nabia/risk/IRiskMeasure.h
        include/nabia/risk/measures/Volatility.h
        include/nabia/risk/measures/ValueAtRisk.h
        include/nabia/risk/measures/ExpectedShortfall.h
        src/risk/IRiskMeasure.cpp
        src/risk/measures/Volatility.cpp
        src/risk/measures/ValueAtRisk.cpp
        src/risk/measures/ExpectedShortfall.cpp
        include/nabia/risk/RiskBudgeting.h
        src/risk/RiskBudgeting.cpp
        include/nabia/signal/IFilter.h
        include/nabia/signal/ButterworthBandPass.h
        include/nabia/signal/EWMA.h
        include/nabia/forecast/IForecaster.h
        include/nabia/forecast/KalmanFilter.h
        include/nabia/forecast/Kernels.h
        include/nabia/forecast/GaussianProcess.h
        include/nabia/dataframe/TimeIndex.h
        include/nabia/dataframe/TimeSeries.h
        include/nabia/dataframe/DataFrame.h
        include/nabia/strategy/PerformanceMetrics.h
        include/nabia/strategy/RiskBudgetingPortfolio.h
)
add_library(nabia::nabia ALIAS nabia)

# PUBLIC: consumers of nabia also see these include dirs / require C++20.
target_include_directories(nabia PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(nabia PUBLIC cxx_std_20)

# PUBLIC: nabia headers expose Eigen types, so consumers need Eigen too.
target_link_libraries(nabia PUBLIC Eigen3::Eigen)
target_link_libraries(nabia PRIVATE Threads::Threads ${NABIA_IIR_TARGET})

# PRIVATE: warnings apply to building nabia itself, not to its consumers.
nabia_set_warnings(nabia)

# --- Subprojects -------------------------------------------------------------
if(NABIA_BUILD_TESTS)
    enable_testing()
endif()

# Before tests/, which adds the application unit tests when the apps are built.
if(NABIA_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(NABIA_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NABIA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NABIA_BUILD_DOCS)
    add_subdirectory(doc)
endif()
