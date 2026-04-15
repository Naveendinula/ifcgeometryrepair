cmake_minimum_required(VERSION 3.24)
project(ifc_geometry_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL CONFIG REQUIRED COMPONENTS Core)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(worker_common STATIC
  src/common.cpp
)

target_link_libraries(worker_common
  PUBLIC
    nlohmann_json::nlohmann_json
)

add_executable(geometry_worker
  src/main.cpp
)

add_executable(shell_worker
  src/shell_worker.cpp
)

target_link_libraries(geometry_worker
  PRIVATE
    worker_common
    CGAL::CGAL
    CGAL::CGAL_Core
)

target_link_libraries(shell_worker
  PRIVATE
    worker_common
    CGAL::CGAL
    CGAL::CGAL_Core
)

if(MSVC)
  target_compile_definitions(worker_common PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(geometry_worker PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(shell_worker PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

function(configure_runtime_output target_name)
  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if(CMAKE_CONFIGURATION_TYPES)
    foreach(config_name ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${config_name}" config_name_upper)
      set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_name_upper} "${CMAKE_BINARY_DIR}"
      )
    endforeach()
  endif()
endfunction()

configure_runtime_output(geometry_worker)
configure_runtime_output(shell_worker)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME geometry_worker_selftest_closed_cube COMMAND geometry_worker --self-test closed_cube)
  add_test(NAME geometry_worker_selftest_inverted_cube COMMAND geometry_worker --self-test inverted_cube)
  add_test(NAME geometry_worker_selftest_duplicate_and_degenerate COMMAND geometry_worker --self-test duplicate_and_degenerate)
  add_test(NAME geometry_worker_selftest_open_mesh COMMAND geometry_worker --self-test open_mesh)
  add_test(NAME geometry_worker_selftest_self_intersection COMMAND geometry_worker --self-test self_intersection)
  add_test(NAME shell_worker_selftest_closed_cube COMMAND shell_worker --self-test closed_cube)
  add_test(NAME shell_worker_selftest_combined_rooms COMMAND shell_worker --self-test combined_rooms)
  add_test(NAME shell_worker_selftest_invalid_contract COMMAND shell_worker --self-test invalid_contract)
  add_test(NAME shell_worker_selftest_degenerate_mesh COMMAND shell_worker --self-test degenerate_mesh)
endif()
