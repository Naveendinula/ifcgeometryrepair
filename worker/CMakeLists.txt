cmake_minimum_required(VERSION 3.24)
project(ifc_geometry_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL CONFIG REQUIRED COMPONENTS Core)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(geometry_worker
  src/main.cpp
)

target_link_libraries(geometry_worker
  PRIVATE
    CGAL::CGAL
    CGAL::CGAL_Core
    nlohmann_json::nlohmann_json
)

if(MSVC)
  target_compile_definitions(geometry_worker PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME geometry_worker_selftest_closed_cube COMMAND geometry_worker --self-test closed_cube)
  add_test(NAME geometry_worker_selftest_inverted_cube COMMAND geometry_worker --self-test inverted_cube)
  add_test(NAME geometry_worker_selftest_duplicate_and_degenerate COMMAND geometry_worker --self-test duplicate_and_degenerate)
  add_test(NAME geometry_worker_selftest_open_mesh COMMAND geometry_worker --self-test open_mesh)
  add_test(NAME geometry_worker_selftest_self_intersection COMMAND geometry_worker --self-test self_intersection)
endif()
