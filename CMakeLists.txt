cmake_minimum_required(VERSION 3.25)

project(nabia
    VERSION 0.1.0
    DESCRIPTION "Numerical library for signal processing and forecasting"
    LANGUAGES CXX)

# --- Options -----------------------------------------------------------------
# Only build tests/examples by default when nabia is the top-level project
# (not when someone pulls it in via add_subdirectory / FetchContent).
option(NABIA_BUILD_TESTS    "Build unit tests"  ${PROJECT_IS_TOP_LEVEL})
option(NABIA_BUILD_EXAMPLES "Build examples"    ${PROJECT_IS_TOP_LEVEL})
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

# --- Library -----------------------------------------------------------------
add_library(nabia
    src/stats/descriptive.cpp
    src/signal/smoothing.cpp
    src/forecast/exponential_smoothing.cpp
    src/optimization/convex/AugmentedLagrangian.cpp
        include/nabia/optimization/IOptimizer.h
        include/nabia/optimization/convex/Constraints.h
        include/nabia/optimization/convex/IConvexOptimization.h
        include/nabia/optimization/convex/AugmentedLagrangian.h
        src/optimization/convex/LBFGS.cpp
        include/nabia/optimization/convex/LBFGS.h
)
add_library(nabia::nabia ALIAS nabia)

# PUBLIC: consumers of nabia also see these include dirs / require C++20.
target_include_directories(nabia PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(nabia PUBLIC cxx_std_20)

# PUBLIC: nabia headers expose Eigen types, so consumers need Eigen too.
target_link_libraries(nabia PUBLIC Eigen3::Eigen)

# PRIVATE: warnings apply to building nabia itself, not to its consumers.
nabia_set_warnings(nabia)

# --- Subprojects -------------------------------------------------------------
if(NABIA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NABIA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NABIA_BUILD_DOCS)
    add_subdirectory(src/optimization/convex/docs)
endif()
