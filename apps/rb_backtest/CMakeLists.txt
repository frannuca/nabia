# nabia_rb_backtest: risk budgeting backtest from a prices CSV and a JSON config.

# nlohmann/json (header-only): an installed copy if there is one, otherwise fetched.
find_package(nlohmann_json 3.11 CONFIG QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
    message(STATUS "nlohmann_json not found, fetching it")
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL    https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        SYSTEM)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Parsing and writing code, shared by the executable and the unit tests.
add_library(rb_backtest_core STATIC config.cpp csv_io.cpp)
target_include_directories(rb_backtest_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rb_backtest_core PUBLIC nabia::nabia PRIVATE nlohmann_json::nlohmann_json)
nabia_set_warnings(rb_backtest_core)

add_executable(nabia_rb_backtest main.cpp)
target_link_libraries(nabia_rb_backtest PRIVATE rb_backtest_core)
nabia_set_warnings(nabia_rb_backtest)

# End-to-end check: run the executable on the bundled example.
if(NABIA_BUILD_TESTS)
    add_test(NAME rb_backtest_example
        COMMAND nabia_rb_backtest
            --prices  ${CMAKE_CURRENT_SOURCE_DIR}/example/prices.csv
            --config  ${CMAKE_CURRENT_SOURCE_DIR}/example/portfolio.json
            --output  ${CMAKE_CURRENT_BINARY_DIR}/example_result.csv
            --metrics ${CMAKE_CURRENT_BINARY_DIR}/example_metrics.csv)
endif()
