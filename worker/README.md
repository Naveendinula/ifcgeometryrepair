# Exact Repair Worker

This directory contains the optional native exact-repair worker used during preprocessing for `IfcSpace` meshes.

## Build

The worker is intended for a Windows-first CMake + vcpkg flow:

```text
cmake -S worker -B worker/build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build worker/build --config Release
ctest --test-dir worker/build -C Release --output-on-failure
```

Required vcpkg packages:

```text
vcpkg install cgal:x64-windows nlohmann-json:x64-windows
```

The resulting binary is expected at `worker/build/geometry_worker.exe`.

## Contract

CLI:

```text
geometry_worker <request.json> <result.json>
```

Request contract version `2`:

- `request.json` contains tessellated `IfcSpace` meshes in meters.
- The worker attempts exact cleanup and regularization per space using CGAL.
- `result.json` contains per-space repaired mesh payloads and topology/repair metadata.

The Python server always orchestrates tessellation and fallback behavior. If the worker is disabled, missing, crashes, or returns malformed output, preprocessing falls back to the in-process Python normalizer and records the fallback in `geometry/repair_report.json`.

## Self Tests

The executable also exposes lightweight self-tests for CTest:

```text
geometry_worker --self-test closed_cube
geometry_worker --self-test inverted_cube
geometry_worker --self-test duplicate_and_degenerate
geometry_worker --self-test open_mesh
geometry_worker --self-test self_intersection
```
