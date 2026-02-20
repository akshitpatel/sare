cmake_minimum_required(VERSION 3.16)
project(sare-hx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic)

# Force macOS SDK path if not found
if(APPLE AND NOT CMAKE_OSX_SYSROOT)
    set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
endif()
if(APPLE)
    include_directories(SYSTEM "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1")
endif()

set(CMAKE_BUILD_TYPE Release)

# ─── Core library ───────────────────────────────────────────────
file(GLOB_RECURSE CORE_SOURCES
    "core/graph/*.cpp"
    "core/energy/*.cpp"
    "core/transforms/*.cpp"
    "core/search/*.cpp"
    "core/verification/*.cpp"
    "core/memory/*.cpp"
    "core/abstraction/*.cpp"
    "core/plasticity/*.cpp"
    "core/causal/*.cpp"
    "core/reflection/*.cpp"
)

# Revert to STATIC library for sare_core
add_library(sare_core STATIC ${CORE_SOURCES})
target_include_directories(sare_core PUBLIC
    ${CMAKE_SOURCE_DIR}/core
)

if(APPLE)
    # Explicitly include C++ standard headers if not found automatically
    target_include_directories(sare_core SYSTEM PUBLIC 
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
    )
endif()

# ─── Tests ──────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    file(GLOB TEST_SOURCES "tests/*.cpp")
    add_executable(sare_tests ${TEST_SOURCES})
    target_link_libraries(sare_tests PRIVATE sare_core GTest::gtest_main) 
    target_include_directories(sare_tests PRIVATE ${CMAKE_SOURCE_DIR}/core)

    include(GoogleTest)
    gtest_discover_tests(sare_tests)

    if(APPLE)
        target_include_directories(sare_tests SYSTEM PRIVATE
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
        )
    endif()
endif()

# ─── PyBind11 bindings ──────────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    
    # pybind11_add_module handles the Python linking
    pybind11_add_module(sare_bindings core/bindings/pybind_module.cpp)
    target_link_libraries(sare_bindings PRIVATE sare_core)
    
    # Force loading of all symbols from static library on macOS
    if(APPLE)
        target_link_options(sare_bindings PRIVATE "LINKER:-all_load")
        target_include_directories(sare_bindings SYSTEM PRIVATE
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
        )
    endif()
endif()
