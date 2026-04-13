# Geometry Worker Scaffold

Phase 3 introduces a stable native-worker boundary without requiring a C++ toolchain at runtime.

Expected CLI contract:

```text
geometry_worker <request.json> <result.json>
```

- `request.json` contains tessellated mesh inputs in meters for `IfcSpace` and `IfcOpeningElement`.
- `result.json` is expected to contain the normalized mesh manifest used by later spatial stages.

The Python backend currently checks for a built worker binary at `worker/build/geometry_worker.exe`.
If that binary is missing, preprocessing falls back to the in-process Python implementation and records
`worker_backend = "python"` in the persisted result artifacts.
