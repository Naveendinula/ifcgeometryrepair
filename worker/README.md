# Native Workers

This directory contains the optional Windows-first native CGAL workers used by the FastAPI app:

- `geometry_worker.exe` for exact `IfcSpace` cleanup during preprocessing
- `shell_worker.exe` for native alpha-wrap shell generation

Both executables are built from the same CMake project and are emitted directly into `worker/build/`.

## Windows Prerequisites

Install the following before configuring the workers:

- Visual Studio Build Tools with the MSVC C++ toolchain
- CMake 3.24 or newer
- `vcpkg`

Required `vcpkg` packages:

```text
vcpkg install cgal:x64-windows nlohmann-json:x64-windows
```

Make sure `VCPKG_ROOT` points to your `vcpkg` checkout. Running the build inside a Developer PowerShell for Visual Studio is the simplest path.

## Build

From the repo root:

```text
cmake -S worker -B worker/build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build worker/build --config Release
ctest --test-dir worker/build -C Release --output-on-failure
```

Expected binaries:

```text
worker/build/geometry_worker.exe
worker/build/shell_worker.exe
```

The CMake project pins `RUNTIME_OUTPUT_DIRECTORY` so both executables land in `worker/build/` even when using a multi-config generator.

## Worker Contracts

`geometry_worker.exe`:

```text
geometry_worker <request.json> <result.json>
```

- request contract version `2`
- input contains tessellated `IfcSpace` meshes in meters
- output contains per-space repaired mesh payloads and topology/repair metadata

`shell_worker.exe`:

```text
shell_worker <request.json> <result.json>
```

- request contract version `1`
- input contains tessellated `IfcSpace` meshes in meters plus alpha-wrap parameters
- output contains the wrapped shell mesh, effective alpha/offset values, backend identity, and worker status

The Python server treats `external_shell_mode="alpha_wrap"` as strict native execution. If `shell_worker.exe` is missing, fails, or returns invalid output, the job fails instead of falling back to heuristic mode.

## Self Tests

`geometry_worker.exe` self-tests:

```text
geometry_worker --self-test closed_cube
geometry_worker --self-test inverted_cube
geometry_worker --self-test duplicate_and_degenerate
geometry_worker --self-test open_mesh
geometry_worker --self-test self_intersection
```

`shell_worker.exe` self-tests:

```text
shell_worker --self-test closed_cube
shell_worker --self-test combined_rooms
shell_worker --self-test invalid_contract
shell_worker --self-test degenerate_mesh
```
