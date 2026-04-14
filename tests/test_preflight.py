from __future__ import annotations

import shutil
from pathlib import Path

from app.mesh_normalizer import normalize_mesh
from app.preflight import run_preflight_validation


def test_preflight_allows_coplanar_shared_boundaries() -> None:
    root = make_preflight_root("touching")
    try:
        left_space = make_space_entity(1, "Room A", normalize_mesh(*build_cube_mesh()))
        right_space = make_space_entity(2, "Room B", normalize_mesh(*build_cube_mesh(origin=(1.0, 0.0, 0.0))))

        result = run_preflight_validation(
            "touching",
            root,
            {"entities": [left_space, right_space]},
            clash_tolerance_m=0.01,
        )

        assert result.payload["status"] == "passed"
        assert result.payload["summary"]["clash_pair_count"] == 0
        assert result.payload["blockers"] == []
        assert (root / "geometry" / "preflight.json").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preflight_flags_volumetric_overlap_between_spaces() -> None:
    root = make_preflight_root("overlap")
    try:
        left_space = make_space_entity(1, "Room A", normalize_mesh(*build_cube_mesh()))
        right_space = make_space_entity(2, "Room B", normalize_mesh(*build_cube_mesh(origin=(0.6, 0.0, 0.0))))

        result = run_preflight_validation(
            "overlap",
            root,
            {"entities": [left_space, right_space]},
            clash_tolerance_m=0.01,
        )

        assert result.payload["status"] == "failed"
        assert result.payload["summary"]["clash_pair_count"] == 1
        assert [blocker["code"] for blocker in result.payload["blockers"]] == ["space_clash"]
        assert result.payload["blockers"][0]["classification"] == "partial_overlap"
        assert result.payload["blockers"][0]["pairs"][0]["detection"] == "triangle_intersection"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preflight_flags_contained_space_without_triangle_crossing() -> None:
    root = make_preflight_root("containment")
    try:
        outer_space = make_space_entity(1, "Outer", normalize_mesh(*build_cube_mesh(size=(2.0, 2.0, 2.0))))
        inner_space = make_space_entity(
            2,
            "Inner",
            normalize_mesh(*build_cube_mesh(origin=(0.5, 0.5, 0.5), size=(0.5, 0.5, 0.5))),
        )

        result = run_preflight_validation(
            "containment",
            root,
            {"entities": [outer_space, inner_space]},
            clash_tolerance_m=0.01,
        )

        assert result.payload["status"] == "failed"
        assert result.payload["summary"]["clash_pair_count"] == 1
        assert result.payload["blockers"][0]["code"] == "space_clash"
        assert result.payload["blockers"][0]["classification"] == "contained_fragment"
        assert result.payload["blockers"][0]["pairs"][0]["detection"] == "containment"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preflight_flags_self_intersecting_space_mesh() -> None:
    root = make_preflight_root("self-intersection")
    try:
        self_intersecting = normalize_mesh(*build_overlapping_cube_components_mesh())
        assert self_intersecting["valid"] is True

        result = run_preflight_validation(
            "self-intersection",
            root,
            {"entities": [make_space_entity(1, "Problem Space", self_intersecting)]},
            clash_tolerance_m=0.01,
        )

        assert result.payload["status"] == "failed"
        assert result.payload["summary"]["self_intersection_space_count"] == 1
        assert [blocker["code"] for blocker in result.payload["blockers"]] == ["self_intersection"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preflight_classifies_exact_duplicate_spaces_with_recommendation() -> None:
    root = make_preflight_root("duplicate")
    try:
        left_space = make_space_entity(1, "Room A", normalize_mesh(*build_cube_mesh()))
        right_space = make_space_entity(2, "Room A Duplicate", normalize_mesh(*build_cube_mesh()))

        result = run_preflight_validation(
            "duplicate",
            root,
            {"entities": [left_space, right_space]},
            clash_tolerance_m=0.01,
        )

        assert result.payload["status"] == "failed"
        assert result.payload["summary"]["clash_group_count"] == 1
        clash_group = result.payload["clash_groups"][0]
        assert clash_group["classification"] == "exact_duplicate"
        assert clash_group["recommended_resolution"]["operation"] == "remove_spaces"
        assert len(clash_group["recommended_resolution"]["spaces_to_remove"]) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preflight_warns_when_auto_repairs_are_present() -> None:
    root = make_preflight_root("repaired")
    try:
        repaired_space = make_space_entity(
            1,
            "Repaired Space",
            normalize_mesh(*build_cube_mesh()),
            repair_actions=["welded_vertices:2"],
        )

        result = run_preflight_validation(
            "repaired",
            root,
            {"entities": [repaired_space]},
            clash_tolerance_m=0.01,
        )

        assert result.payload["status"] == "passed"
        assert result.payload["summary"]["warning_count"] == 1
        assert [warning["code"] for warning in result.payload["warnings"]] == ["auto_repair_applied"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def make_space_entity(
    express_id: int,
    name: str,
    mesh_result: dict,
    *,
    repair_actions: list[str] | None = None,
) -> dict:
    payload = dict(mesh_result)
    payload["repair_actions"] = repair_actions if repair_actions is not None else list(mesh_result.get("repair_actions", []))
    return {
        "object_name": f"space_{express_id}",
        "global_id": f"space-{express_id}",
        "express_id": express_id,
        "name": name,
        "entity_type": "IfcSpace",
        "has_representation": True,
        **payload,
    }


def build_overlapping_cube_components_mesh() -> tuple[list[list[float]], list[list[int]]]:
    left_vertices, left_faces = build_cube_mesh()
    right_vertices, right_faces = build_cube_mesh(origin=(0.6, 0.0, 0.0))
    return merge_meshes((left_vertices, left_faces), (right_vertices, right_faces))


def build_cube_mesh(
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[list[list[float]], list[list[int]]]:
    origin_x, origin_y, origin_z = origin
    size_x, size_y, size_z = size
    vertices = [
        [origin_x, origin_y, origin_z],
        [origin_x + size_x, origin_y, origin_z],
        [origin_x + size_x, origin_y + size_y, origin_z],
        [origin_x, origin_y + size_y, origin_z],
        [origin_x, origin_y, origin_z + size_z],
        [origin_x + size_x, origin_y, origin_z + size_z],
        [origin_x + size_x, origin_y + size_y, origin_z + size_z],
        [origin_x, origin_y + size_y, origin_z + size_z],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [0, 7, 3],
        [0, 4, 7],
        [3, 6, 2],
        [3, 7, 6],
    ]
    return vertices, faces


def merge_meshes(
    left_mesh: tuple[list[list[float]], list[list[int]]],
    right_mesh: tuple[list[list[float]], list[list[int]]],
) -> tuple[list[list[float]], list[list[int]]]:
    left_vertices, left_faces = left_mesh
    right_vertices, right_faces = right_mesh
    vertex_offset = len(left_vertices)
    return (
        left_vertices + right_vertices,
        left_faces + [[index + vertex_offset for index in face] for face in right_faces],
    )


def make_preflight_root(label: str) -> Path:
    root = Path("jobs") / "test_runs" / f"preflight-{label}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=False)
    return root
