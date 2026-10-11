# Python extension module nabia._nabia (pybind11).
#
# Development build: the module is written into python/nabia/ next to the
# Python sources, so `PYTHONPATH=python python -c "import nabia"` works:
#   cmake --preset python && cmake --build --preset python
# Wheel build (pip install ./python, via scikit-build-core): installed into the
# nabia package of the wheel.

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

pybind11_add_module(_nabia src/nabia_module.cpp)
target_link_libraries(_nabia PRIVATE rb_backtest_core)
nabia_set_warnings(_nabia)

if(SKBUILD)
    install(TARGETS _nabia LIBRARY DESTINATION nabia)
else()
    set_target_properties(_nabia PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nabia)
    if(NABIA_BUILD_TESTS)
        add_test(NAME python_tests
            COMMAND "${Python_EXECUTABLE}" -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(python_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()
