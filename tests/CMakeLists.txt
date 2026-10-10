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
)
target_link_libraries(nabia_tests PRIVATE nabia::nabia Catch2::Catch2WithMain)
nabia_set_warnings(nabia_tests)

# Register each TEST_CASE individually with CTest.
include(Catch)
catch_discover_tests(nabia_tests)
