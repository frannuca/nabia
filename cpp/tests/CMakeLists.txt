include(FetchContent)

# Download Catch2 at configure time (cached in build/_deps afterwards).
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
    SYSTEM)  # treat Catch2 headers as system headers -> no warnings from them
FetchContent_MakeAvailable(Catch2)

add_executable(nabia_tests
    test_descriptive.cpp
    test_smoothing.cpp
    test_exponential_smoothing.cpp
    test_augmented_lagrangian.cpp
    test_lbfgs.cpp
    test_sqp.cpp
    test_nongradopt.cpp
    test_filters.cpp
    test_forecasters.cpp
    test_dataframe.cpp
    test_risk_budgeting_portfolio.cpp
    test_risk_measures.cpp
    test_risk_budgeting.cpp
    test_regression.cpp
)
target_link_libraries(nabia_tests PRIVATE nabia::nabia Catch2::Catch2WithMain)
nabia_set_warnings(nabia_tests)

# Register each TEST_CASE individually with CTest.
include(Catch)
catch_discover_tests(nabia_tests)

# Unit tests of the command-line application's parsing and output code.
if(TARGET rb_backtest_core)
    target_sources(nabia_tests PRIVATE test_rb_backtest_app.cpp)
    target_link_libraries(nabia_tests PRIVATE rb_backtest_core)
endif()
