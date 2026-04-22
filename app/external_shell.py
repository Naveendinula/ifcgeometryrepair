from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union

from .mesh_normalizer import build_obj_text


FACE_AREA_EPSILON = 1e-9
NORMAL_DEGREES = 5.0
NORMAL_DOT_THRESHOLD = float(np.cos(np.deg2rad(NORMAL_DEGREES)))
PLANE_OFFSET_TOLERANCE_M = 1e-4
ORIENTATION_Z_THRESHOLD = 0.707
DEFAULT_SHELL_MATCH_TOLERANCE_M = 0.05
MIN_OVERLAP_AREA_M2 = 1e-6
ALPHA_WRAP_CONTRACT_VERSION = 1
ALPHA_WRAP_EPSILON = 1e-3
ALPHA_WRAP_ALPHA_BUFFER_M = 0.05
ALPHA_WRAP_NATIVE_BACKEND = "cpp-cgal-alpha-wrap"
AABB_LEAF_TRIANGLE_COUNT = 16
SURFACE_CLASSES = (
    "external_wall",
    "roof",
    "ground_floor",
    "internal_void",
    "unclassified",
)


@dataclass(slots=True)
class ExternalShellResult:
    payload: dict[str, Any]


@dataclass(slots=True)
class SurfacePatch:
    surface_id: str
    object_name: str
    space_global_id: str | None
    space_express_id: int | None
    space_name: str | None
    classification: str
    area_m2: float
    normal: np.ndarray
    centroid: np.ndarray
    plane_point: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    union_polygon_2d: Polygon | MultiPolygon
    triangles_3d: list[np.ndarray]
    mesh: dict[str, Any]
    reason: str | None = None
    artifacts: dict[str, str | None] | None = None
    source_surface_id: str | None = None
    subtracted_internal_surface_ids: list[str] | None = None


@dataclass(slots=True)
class WrapTriangle:
    triangle_id: str
    vertices: np.ndarray
    normal: np.ndarray
    plane_point: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray


@dataclass(slots=True)
class TriangleAABBNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left: TriangleAABBNode | None
    right: TriangleAABBNode | None
    triangles: list[WrapTriangle]


def run_external_shell_classification(
    job_id: str,
    job_dir: Path,
    preprocessing_result: dict[str, Any],
    external_candidates_result: dict[str, Any],
    *,
    mode_requested: str = "alpha_wrap",
    worker_binary: Path | None = None,
    thickness_threshold_m: float,
    alpha_m_requested: float,
    offset_m_requested: float,
) -> ExternalShellResult:
    external_shell_dir = job_dir / "geometry" / "external_shell"
    classes_dir = external_shell_dir / "classes"
    external_shell_dir.mkdir(parents=True, exist_ok=True)
    classes_dir.mkdir(parents=True, exist_ok=True)

    space_entities = sorted(
        [entity for entity in preprocessing_result.get("entities", []) if entity.get("entity_type") == "IfcSpace"],
        key=_entity_sort_key,
    )
    valid_space_entities = [
        entity
        for entity in space_entities
        if entity.get("valid") and entity.get("mesh") and entity["mesh"].get("faces")
    ]
    skipped_spaces = [
        {
            "global_id": entity.get("global_id"),
            "express_id": entity["express_id"],
            "name": entity.get("name"),
            "reason": entity.get("reason") or "Space mesh unavailable",
        }
        for entity in space_entities
        if entity not in valid_space_entities
    ]

    candidate_surfaces = _load_candidate_surfaces(external_candidates_result)
    effective_alpha_m = max(float(alpha_m_requested), float(thickness_threshold_m) + ALPHA_WRAP_ALPHA_BUFFER_M)
    effective_offset_m = float(offset_m_requested)
    request_payload = {
        "contract_version": ALPHA_WRAP_CONTRACT_VERSION,
        "job_id": job_id,
        "unit": "meter",
        "mode_requested": mode_requested,
        "epsilon": ALPHA_WRAP_EPSILON,
        "alpha_m_requested": float(alpha_m_requested),
        "alpha_m_effective": float(effective_alpha_m),
        "offset_m_requested": float(offset_m_requested),
        "offset_m_effective": float(effective_offset_m),
        "space_meshes": [
            {
                "global_id": entity.get("global_id"),
                "express_id": entity["express_id"],
                "name": entity.get("name"),
                "mesh": entity.get("mesh"),
            }
            for entity in valid_space_entities
        ],
        "candidate_surfaces": [
            {
                "surface_id": surface.surface_id,
                "space_global_id": surface.space_global_id,
                "space_express_id": surface.space_express_id,
                "space_name": surface.space_name,
                "area_m2": round(float(surface.area_m2), 6),
                "normal": _round_vector(surface.normal),
                "centroid": _round_vector(surface.centroid),
            }
            for surface in candidate_surfaces
        ],
    }
    _write_json(external_shell_dir / "request.json", request_payload)

    shell_started = time.perf_counter()
    fallback_reason = None
    mode_effective = mode_requested
    if mode_requested == "alpha_wrap":
        if not valid_space_entities and not candidate_surfaces:
            shell_payload = {
                "backend": "none",
                "status": "idle",
                "alpha_m_effective": float(effective_alpha_m),
                "offset_m_effective": float(effective_offset_m),
                "shell_mesh": {"vertices": [], "faces": []},
            }
        else:
            shell_payload = generate_alpha_wrap_shell(request_payload, worker_binary, external_shell_dir)
    else:
        shell_payload = generate_heuristic_shell(request_payload)
    shell_generation_time_ms = (time.perf_counter() - shell_started) * 1000.0

    shell_mesh = shell_payload.get("shell_mesh") or {"vertices": [], "faces": []}
    shell_patches = _extract_surface_patches_from_mesh(
        shell_mesh,
        id_prefix="shell",
        space_global_id=None,
        space_express_id=None,
        space_name="Envelope Shell",
    )

    classification_started = time.perf_counter()
    if mode_effective == "alpha_wrap":
        shell_match_count, internal_void_count, unclassified_count = _classify_with_alpha_wrap(
            candidate_surfaces,
            shell_mesh,
            offset_tolerance_m=float(shell_payload.get("offset_m_effective", effective_offset_m)) + ALPHA_WRAP_EPSILON,
        )
    else:
        shell_match_count, internal_void_count, unclassified_count = _classify_with_heuristic_shell(
            candidate_surfaces,
            shell_patches,
            shell_payload,
        )
    classification_time_ms = (time.perf_counter() - classification_started) * 1000.0

    artifacts = {
        "request": "geometry/external_shell/request.json",
        "result": "geometry/external_shell/result.json",
        "shell_obj": "geometry/external_shell/shell.obj",
        "surfaces_all": "geometry/external_shell/surfaces_all.obj",
        "classes": {
            surface_class: f"geometry/external_shell/classes/{surface_class}.obj"
            for surface_class in SURFACE_CLASSES
        },
    }

    for surface in candidate_surfaces:
        class_obj = artifacts["classes"][surface.classification]
        surface.artifacts = {
            "classified_obj": artifacts["surfaces_all"],
            "class_obj": class_obj,
            "shell_obj": artifacts["shell_obj"],
        }

    _write_surface_artifacts(job_dir, artifacts, shell_mesh, candidate_surfaces)

    per_class_counts = {surface_class: 0 for surface_class in SURFACE_CLASSES}
    per_class_area_m2 = {surface_class: 0.0 for surface_class in SURFACE_CLASSES}
    for surface in candidate_surfaces:
        per_class_counts[surface.classification] += 1
        per_class_area_m2[surface.classification] += float(surface.area_m2)

    shell_mesh_stats = {
        "vertex_count": int(len(shell_mesh.get("vertices", []))),
        "face_count": int(len(shell_mesh.get("faces", []))),
        "surface_patch_count": len(shell_patches),
    }
    if "aabb_min" in shell_payload and "aabb_max" in shell_payload:
        shell_mesh_stats["aabb_min"] = _round_vector(shell_payload["aabb_min"])
        shell_mesh_stats["aabb_max"] = _round_vector(shell_payload["aabb_max"])

    summary = {
        "processed_space_count": len(valid_space_entities),
        "skipped_space_count": len(skipped_spaces),
        "candidate_surface_count": len(candidate_surfaces),
        "internal_partition_match_count": 0,
        "shell_match_count": shell_match_count,
        "internal_void_count": internal_void_count,
        "unclassified_count": unclassified_count,
        "per_class_counts": per_class_counts,
        "per_class_area_m2": {
            surface_class: round(float(area_m2), 6)
            for surface_class, area_m2 in per_class_area_m2.items()
        },
        "total_classified_area_m2": round(
            float(sum(surface.area_m2 for surface in candidate_surfaces if surface.classification != "unclassified")),
            6,
        ),
        "shell_generation_time_ms": round(float(shell_generation_time_ms), 3),
        "classification_time_ms": round(float(classification_time_ms), 3),
    }
    alpha_wrap_payload = {
        "status": (
            shell_payload.get("status", "ok")
            if mode_effective == "alpha_wrap"
            else "compatibility_heuristic"
        ),
        "backend": shell_payload.get("backend", "python"),
        "epsilon": float(ALPHA_WRAP_EPSILON),
        "alpha_m_requested": round(float(alpha_m_requested), 6),
        "alpha_m_effective": round(float(shell_payload.get("alpha_m_effective", effective_alpha_m)), 6),
        "offset_m_requested": round(float(offset_m_requested), 6),
        "offset_m_effective": round(float(shell_payload.get("offset_m_effective", effective_offset_m)), 6),
        "generation_time_ms": round(float(shell_generation_time_ms), 3),
        "classification_time_ms": round(float(classification_time_ms), 3),
        "triangle_count": int(len(shell_mesh.get("faces", []))),
        "vertex_count": int(len(shell_mesh.get("vertices", []))),
        "error": None,
    }

    result_payload = {
        "job_id": job_id,
        "mode_requested": mode_requested,
        "mode_effective": mode_effective,
        "fallback_reason": fallback_reason,
        "shell_backend": shell_payload.get("backend", "python"),
        "shell_generation_time_ms": round(float(shell_generation_time_ms), 3),
        "classification_time_ms": round(float(classification_time_ms), 3),
        "shell_mesh_stats": shell_mesh_stats,
        "alpha_wrap": alpha_wrap_payload,
        "summary": summary,
        "skipped_spaces": skipped_spaces,
        "surfaces": [_serialize_surface(surface) for surface in candidate_surfaces],
        "artifacts": artifacts,
    }
    _write_json(external_shell_dir / "result.json", result_payload)
    return ExternalShellResult(payload=result_payload)


def generate_alpha_wrap_shell(
    request_payload: dict[str, Any],
    worker_binary: Path | None,
    workspace_dir: Path,
) -> dict[str, Any]:
    if worker_binary is None or not worker_binary.exists():
        raise RuntimeError("Native alpha-wrap shell worker is unavailable")
    worker_temp_root = workspace_dir / "_worker"
    worker_temp_root.mkdir(parents=True, exist_ok=True)
    request_path = worker_temp_root / "request.json"
    result_path = worker_temp_root / "result.json"
    _write_json(request_path, request_payload)
    if result_path.exists():
        result_path.unlink()
    _invoke_alpha_wrap_worker(worker_binary, request_path, result_path)
    return _parse_alpha_wrap_response(_read_json(result_path), request_payload)


def _invoke_alpha_wrap_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
    command = [str(worker_binary), str(request_path), str(result_path)]
    if worker_binary.suffix.lower() == ".py":
        command = [sys.executable, str(worker_binary), str(request_path), str(result_path)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "alpha-wrap worker failed"
        raise RuntimeError(stderr)


def _parse_alpha_wrap_response(payload: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("contract_version") != ALPHA_WRAP_CONTRACT_VERSION:
        raise RuntimeError("Alpha-wrap worker returned an unsupported contract version")
    if payload.get("status") != "ok":
        raise RuntimeError(payload.get("reason") or "Alpha-wrap worker failed")

    shell_mesh = payload.get("shell_mesh") or {"vertices": [], "faces": []}
    faces = shell_mesh.get("faces", [])
    vertices = shell_mesh.get("vertices", [])
    if request_payload.get("space_meshes") and (not vertices or not faces):
        raise RuntimeError("Alpha-wrap worker returned an empty shell mesh")

    parsed_payload = {
        "backend": payload.get("backend", ALPHA_WRAP_NATIVE_BACKEND),
        "status": "ok",
        "alpha_m_effective": float(payload.get("alpha_m_effective", request_payload["alpha_m_effective"])),
        "offset_m_effective": float(payload.get("offset_m_effective", request_payload["offset_m_effective"])),
        "shell_mesh": shell_mesh,
    }
    if "generation_time_ms" in payload:
        parsed_payload["generation_time_ms"] = payload["generation_time_ms"]
    if "reason" in payload and payload.get("reason"):
        parsed_payload["reason"] = payload["reason"]
    return parsed_payload


def generate_heuristic_shell(request_payload: dict[str, Any]) -> dict[str, Any]:
    vertices: list[list[float]] = []
    for space in request_payload.get("space_meshes", []):
        mesh = space.get("mesh") or {}
        vertices.extend(mesh.get("vertices", []))

    if not vertices:
        return {
            "backend": "python",
            "shell_mesh": {"vertices": [], "faces": []},
            "match_tolerance_m": DEFAULT_SHELL_MATCH_TOLERANCE_M,
        }

    vertices_array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    aabb_min = np.min(vertices_array, axis=0)
    aabb_max = np.max(vertices_array, axis=0)
    shell_mesh = _build_box_mesh(aabb_min, aabb_max)
    return {
        "backend": "python",
        "shell_mesh": shell_mesh,
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "match_tolerance_m": DEFAULT_SHELL_MATCH_TOLERANCE_M,
    }


def _classify_with_alpha_wrap(
    candidate_surfaces: list[SurfacePatch],
    shell_mesh: dict[str, Any],
    *,
    offset_tolerance_m: float,
) -> tuple[int, int, int]:
    wrap_triangles = _load_wrap_triangles(shell_mesh)
    triangle_index = _build_triangle_aabb_tree(wrap_triangles)
    shell_match_count = 0
    internal_void_count = 0

    for surface in candidate_surfaces:
        triangle = _find_alpha_wrap_hit(surface, triangle_index, offset_tolerance_m=offset_tolerance_m)
        if triangle is not None:
            surface.classification = _surface_class_from_normal(surface.normal)
            surface.reason = f"alpha_wrap_hit:{triangle.triangle_id}"
            shell_match_count += 1
            continue

        surface.classification = "internal_void"
        surface.reason = "no_alpha_wrap_hit"
        internal_void_count += 1

    return shell_match_count, internal_void_count, 0


def _classify_with_heuristic_shell(
    candidate_surfaces: list[SurfacePatch],
    shell_patches: list[SurfacePatch],
    shell_payload: dict[str, Any],
) -> tuple[int, int, int]:
    shell_match_count = 0
    internal_void_count = 0
    unclassified_count = 0
    shell_match_tolerance_m = float(shell_payload.get("match_tolerance_m", DEFAULT_SHELL_MATCH_TOLERANCE_M))

    for surface in candidate_surfaces:
        shell_match = _best_overlap_match(surface, shell_patches, plane_tolerance_m=shell_match_tolerance_m)
        if shell_match is not None:
            surface.classification = _surface_class_from_normal(surface.normal)
            surface.reason = f"shell_match:{shell_match.surface_id}"
            shell_match_count += 1
            continue

        internal_void_reason = _classify_internal_void(surface, shell_payload)
        if internal_void_reason:
            surface.classification = "internal_void"
            surface.reason = internal_void_reason
            internal_void_count += 1
            continue

        surface.classification = "unclassified"
        surface.reason = "No matching shell surface"
        unclassified_count += 1

    return shell_match_count, internal_void_count, unclassified_count


def _load_wrap_triangles(shell_mesh: dict[str, Any]) -> list[WrapTriangle]:
    vertices = np.asarray(shell_mesh.get("vertices", []), dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(shell_mesh.get("faces", []), dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return []

    triangles = vertices[faces]
    wrap_triangles: list[WrapTriangle] = []
    for triangle_index, triangle in enumerate(triangles):
        cross_product = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        magnitude = float(np.linalg.norm(cross_product))
        if magnitude <= FACE_AREA_EPSILON:
            continue
        normal = cross_product / magnitude
        wrap_triangles.append(
            WrapTriangle(
                triangle_id=f"aw_{triangle_index}",
                vertices=np.asarray(triangle, dtype=np.float64),
                normal=np.asarray(normal, dtype=np.float64),
                plane_point=np.asarray(triangle[0], dtype=np.float64),
                aabb_min=np.min(triangle, axis=0),
                aabb_max=np.max(triangle, axis=0),
            )
        )
    return wrap_triangles


def _build_triangle_aabb_tree(triangles: list[WrapTriangle]) -> TriangleAABBNode | None:
    if not triangles:
        return None
    aabb_min = np.min(np.stack([triangle.aabb_min for triangle in triangles]), axis=0)
    aabb_max = np.max(np.stack([triangle.aabb_max for triangle in triangles]), axis=0)
    if len(triangles) <= AABB_LEAF_TRIANGLE_COUNT:
        return TriangleAABBNode(aabb_min=aabb_min, aabb_max=aabb_max, left=None, right=None, triangles=triangles)

    extents = aabb_max - aabb_min
    axis = int(np.argmax(extents))
    sorted_triangles = sorted(
        triangles,
        key=lambda triangle: float((triangle.aabb_min[axis] + triangle.aabb_max[axis]) / 2.0),
    )
    midpoint = len(sorted_triangles) // 2
    left = _build_triangle_aabb_tree(sorted_triangles[:midpoint])
    right = _build_triangle_aabb_tree(sorted_triangles[midpoint:])
    return TriangleAABBNode(aabb_min=aabb_min, aabb_max=aabb_max, left=left, right=right, triangles=[])


def _find_alpha_wrap_hit(
    surface: SurfacePatch,
    triangle_index: TriangleAABBNode | None,
    *,
    offset_tolerance_m: float,
) -> WrapTriangle | None:
    candidate_min, candidate_max = _surface_aabb(surface, padding=offset_tolerance_m)
    for triangle in _query_triangle_aabb_tree(triangle_index, candidate_min, candidate_max):
        if _surface_matches_wrap_triangle(surface, triangle, offset_tolerance_m=offset_tolerance_m):
            return triangle
    return None


def _query_triangle_aabb_tree(
    node: TriangleAABBNode | None,
    query_min: np.ndarray,
    query_max: np.ndarray,
) -> list[WrapTriangle]:
    if node is None or not _aabb_overlap(node.aabb_min, node.aabb_max, query_min, query_max):
        return []
    if node.left is None and node.right is None:
        return [
            triangle
            for triangle in node.triangles
            if _aabb_overlap(triangle.aabb_min, triangle.aabb_max, query_min, query_max)
        ]
    return [
        *_query_triangle_aabb_tree(node.left, query_min, query_max),
        *_query_triangle_aabb_tree(node.right, query_min, query_max),
    ]


def _surface_matches_wrap_triangle(
    surface: SurfacePatch,
    triangle: WrapTriangle,
    *,
    offset_tolerance_m: float,
) -> bool:
    if abs(float(np.dot(surface.normal, triangle.normal))) <= 1.0 - ALPHA_WRAP_EPSILON:
        return False
    plane_distances = [
        abs(float(np.dot(surface.normal, vertex - surface.plane_point)))
        for vertex in triangle.vertices
    ]
    if max(plane_distances, default=0.0) > offset_tolerance_m:
        return False
    overlap_area = _triangle_overlap_area_on_surface(surface, triangle)
    return overlap_area >= MIN_OVERLAP_AREA_M2


def _triangle_overlap_area_on_surface(surface: SurfacePatch, triangle: WrapTriangle) -> float:
    projected_triangle = triangle.vertices - np.outer(
        np.einsum("ij,j->i", triangle.vertices - surface.plane_point, surface.normal),
        surface.normal,
    )
    triangle_polygon = _triangle_polygon(
        _project_triangle(projected_triangle, surface.plane_point, surface.basis_u, surface.basis_v)
    )
    if triangle_polygon is None:
        return 0.0
    overlap = surface.union_polygon_2d.intersection(triangle_polygon)
    return float(sum(polygon.area for polygon in _extract_polygons(overlap)))


def _surface_aabb(surface: SurfacePatch, *, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if surface.mesh.get("vertices"):
        vertices = np.asarray(surface.mesh["vertices"], dtype=np.float64).reshape(-1, 3)
    else:
        vertices = np.vstack(surface.triangles_3d) if surface.triangles_3d else np.zeros((0, 3), dtype=np.float64)
    if len(vertices) == 0:
        zero = np.zeros(3, dtype=np.float64)
        return zero, zero
    delta = np.full(3, float(padding), dtype=np.float64)
    return np.min(vertices, axis=0) - delta, np.max(vertices, axis=0) + delta


def _aabb_overlap(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
) -> bool:
    return bool(np.all(left_min <= right_max) and np.all(right_min <= left_max))


def _extract_space_surfaces(space_entities: list[dict[str, Any]]) -> list[SurfacePatch]:
    patches: list[SurfacePatch] = []
    for space_index, entity in enumerate(sorted(space_entities, key=_entity_sort_key)):
        extracted = _extract_surface_patches_from_mesh(
            entity["mesh"],
            id_prefix=f"surf_{space_index}",
            space_global_id=entity.get("global_id"),
            space_express_id=entity["express_id"],
            space_name=entity.get("name"),
        )
        patches.extend(extracted)
    return patches


def _load_candidate_surfaces(external_candidates_result: dict[str, Any]) -> list[SurfacePatch]:
    candidate_surfaces: list[tuple[tuple[float, ...], SurfacePatch]] = []
    for surface in external_candidates_result.get("candidate_surfaces", []):
        patch = _candidate_surface_to_patch(surface)
        if patch is None:
            continue
        sort_key = (
            round(float(patch.centroid[0]), 6),
            round(float(patch.centroid[1]), 6),
            round(float(patch.centroid[2]), 6),
            round(float(patch.normal[0]), 6),
            round(float(patch.normal[1]), 6),
            round(float(patch.normal[2]), 6),
            patch.surface_id,
        )
        candidate_surfaces.append((sort_key, patch))
    return [patch for _, patch in sorted(candidate_surfaces, key=lambda item: item[0])]


def _candidate_surface_to_patch(surface: dict[str, Any]) -> SurfacePatch | None:
    normal = np.asarray(surface.get("plane_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= FACE_AREA_EPSILON:
        return None
    normal /= magnitude
    plane_point = np.asarray(surface.get("plane_point", [0.0, 0.0, 0.0]), dtype=np.float64)
    basis_u, basis_v = _plane_basis(normal)
    triangle_arrays = [np.asarray(triangle, dtype=np.float64) for triangle in surface.get("triangles", [])]
    if not triangle_arrays:
        return None
    union_polygon = _polygon_from_rings_3d(surface.get("polygon_rings_3d", []), plane_point, basis_u, basis_v)
    if union_polygon is None:
        polygons = [
            polygon
            for triangle in triangle_arrays
            if (polygon := _triangle_polygon(_project_triangle(triangle, plane_point, basis_u, basis_v))) is not None
        ]
        if not polygons:
            return None
        union_polygon = unary_union(polygons)
    centroid = np.asarray(surface.get("centroid") or _lift_point(union_polygon.centroid.coords[0], plane_point, basis_u, basis_v), dtype=np.float64)
    mesh = _triangles_to_mesh([triangle.tolist() for triangle in triangle_arrays])
    return SurfacePatch(
        surface_id=surface["surface_id"],
        object_name=surface.get("object_name") or surface["surface_id"],
        space_global_id=surface.get("space_global_id"),
        space_express_id=surface.get("space_express_id"),
        space_name=surface.get("space_name"),
        classification="unclassified",
        area_m2=float(surface.get("area_m2", union_polygon.area)),
        normal=normal,
        centroid=centroid,
        plane_point=plane_point,
        basis_u=basis_u,
        basis_v=basis_v,
        union_polygon_2d=union_polygon,
        triangles_3d=triangle_arrays,
        mesh=mesh,
        source_surface_id=surface.get("source_surface_id"),
        subtracted_internal_surface_ids=list(surface.get("subtracted_internal_surface_ids", [])),
    )


def _extract_surface_patches_from_mesh(
    mesh: dict[str, Any],
    *,
    id_prefix: str,
    space_global_id: str | None,
    space_express_id: int | None,
    space_name: str | None,
) -> list[SurfacePatch]:
    vertices = np.asarray(mesh.get("vertices", []), dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.get("faces", []), dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return []

    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    magnitudes = np.linalg.norm(cross_products, axis=1)
    valid_mask = magnitudes > FACE_AREA_EPSILON
    if not np.any(valid_mask):
        return []

    faces = faces[valid_mask]
    triangles = triangles[valid_mask]
    normals = cross_products[valid_mask] / magnitudes[valid_mask][:, np.newaxis]
    centroids = np.mean(triangles, axis=1)
    plane_offsets = np.einsum("ij,ij->i", normals, centroids)

    adjacency = _build_face_adjacency(faces)
    visited = np.zeros(len(faces), dtype=bool)
    groups: list[list[int]] = []
    for face_index in range(len(faces)):
        if visited[face_index]:
            continue
        queue = [face_index]
        visited[face_index] = True
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                if np.dot(normals[current], normals[neighbor]) < NORMAL_DOT_THRESHOLD:
                    continue
                if abs(float(plane_offsets[neighbor] - plane_offsets[current])) > PLANE_OFFSET_TOLERANCE_M:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        groups.append(component)

    surface_candidates: list[tuple[tuple[float, ...], SurfacePatch]] = []
    for _, face_indices in enumerate(groups):
        group_normals = normals[face_indices]
        average_normal = np.mean(group_normals, axis=0)
        average_normal /= np.linalg.norm(average_normal)
        plane_offset = float(np.mean(np.einsum("ij,j->i", centroids[face_indices], average_normal)))
        plane_point = average_normal * plane_offset
        basis_u, basis_v = _plane_basis(average_normal)

        polygons: list[Polygon] = []
        for face_index in face_indices:
            polygon = _triangle_polygon(_project_triangle(triangles[face_index], plane_point, basis_u, basis_v))
            if polygon is not None:
                polygons.append(polygon)
        if not polygons:
            continue

        unioned = unary_union(polygons)
        extracted_polygons = _extract_polygons(unioned)
        if not extracted_polygons:
            continue

        triangles_3d = _triangulate_polygons_3d(extracted_polygons, plane_point, basis_u, basis_v)
        if not triangles_3d:
            continue

        union_polygon = unary_union(extracted_polygons)
        area_m2 = float(sum(polygon.area for polygon in extracted_polygons))
        centroid = _lift_point(union_polygon.centroid.coords[0], plane_point, basis_u, basis_v)
        mesh_payload = _triangles_to_mesh(triangles_3d)
        placeholder = SurfacePatch(
            surface_id="",
            object_name="",
            space_global_id=space_global_id,
            space_express_id=space_express_id,
            space_name=space_name,
            classification="unclassified",
            area_m2=area_m2,
            normal=average_normal,
            centroid=np.asarray(centroid, dtype=np.float64),
            plane_point=plane_point,
            basis_u=basis_u,
            basis_v=basis_v,
            union_polygon_2d=union_polygon,
            triangles_3d=[np.asarray(triangle, dtype=np.float64) for triangle in triangles_3d],
            mesh=mesh_payload,
        )
        sort_key = (
            round(float(placeholder.centroid[0]), 6),
            round(float(placeholder.centroid[1]), 6),
            round(float(placeholder.centroid[2]), 6),
            round(float(placeholder.normal[0]), 6),
            round(float(placeholder.normal[1]), 6),
            round(float(placeholder.normal[2]), 6),
        )
        surface_candidates.append((sort_key, placeholder))

    patches: list[SurfacePatch] = []
    for patch_index, (_, patch) in enumerate(sorted(surface_candidates, key=lambda item: item[0])):
        surface_id = f"{id_prefix}_{patch_index}"
        patch.surface_id = surface_id
        patch.object_name = surface_id
        patches.append(patch)
    return patches


def _polygon_from_rings_3d(
    rings_3d: list[list[list[float]]],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> Polygon | MultiPolygon | None:
    if not rings_3d:
        return None
    shell = _project_ring_to_basis(rings_3d[0], plane_point, basis_u, basis_v)
    holes = [_project_ring_to_basis(ring, plane_point, basis_u, basis_v) for ring in rings_3d[1:]]
    polygon = Polygon(shell, holes)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if isinstance(polygon, (Polygon, MultiPolygon)):
        return polygon
    extracted = _extract_polygons(polygon)
    if not extracted:
        return None
    return extracted[0] if len(extracted) == 1 else unary_union(extracted)


def _project_ring_to_basis(
    ring_3d: list[list[float]],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[tuple[float, float]]:
    projected: list[tuple[float, float]] = []
    for point in ring_3d:
        point_array = np.asarray(point, dtype=np.float64)
        relative = point_array - plane_point
        projected.append((float(relative @ basis_u), float(relative @ basis_v)))
    return projected


def _best_overlap_match(
    source_surface: SurfacePatch,
    candidate_targets: list[SurfacePatch],
    *,
    plane_tolerance_m: float,
    same_direction: bool = False,
) -> SurfacePatch | None:
    best_match = None
    best_overlap = 0.0
    for target in candidate_targets:
        normal_alignment = float(np.dot(source_surface.normal, target.normal))
        if same_direction:
            if normal_alignment < NORMAL_DOT_THRESHOLD:
                continue
        elif abs(normal_alignment) < NORMAL_DOT_THRESHOLD:
            continue

        plane_distance = abs(float(np.dot(target.normal, source_surface.centroid - target.plane_point)))
        if plane_distance > plane_tolerance_m:
            continue

        overlap_area = _surface_overlap_area(source_surface, target)
        if overlap_area < MIN_OVERLAP_AREA_M2:
            continue

        if overlap_area > best_overlap:
            best_overlap = overlap_area
            best_match = target
    return best_match


def _surface_overlap_area(source_surface: SurfacePatch, target_surface: SurfacePatch) -> float:
    projected_polygons: list[Polygon] = []
    for triangle in source_surface.triangles_3d:
        polygon = _triangle_polygon(
            _project_triangle(triangle, target_surface.plane_point, target_surface.basis_u, target_surface.basis_v)
        )
        if polygon is not None:
            projected_polygons.append(polygon)
    if not projected_polygons:
        return 0.0

    projected_union = unary_union(projected_polygons)
    overlap = projected_union.intersection(target_surface.union_polygon_2d)
    return float(sum(polygon.area for polygon in _extract_polygons(overlap)))


def _classify_internal_void(surface: SurfacePatch, shell_payload: dict[str, Any]) -> str | None:
    aabb_min = shell_payload.get("aabb_min")
    aabb_max = shell_payload.get("aabb_max")
    if aabb_min is None or aabb_max is None:
        return None

    lower = np.asarray(aabb_min, dtype=np.float64)
    upper = np.asarray(aabb_max, dtype=np.float64)
    centroid = surface.centroid
    tolerance = DEFAULT_SHELL_MATCH_TOLERANCE_M * 2.0
    inside = np.all(centroid > (lower + tolerance)) and np.all(centroid < (upper - tolerance))
    if inside:
        return "Candidate surface lies inside the shell volume"
    return None


def _surface_class_from_normal(normal: np.ndarray) -> str:
    if float(normal[2]) >= ORIENTATION_Z_THRESHOLD:
        return "roof"
    if float(normal[2]) <= -ORIENTATION_Z_THRESHOLD:
        return "ground_floor"
    return "external_wall"


def _write_surface_artifacts(
    job_dir: Path,
    artifacts: dict[str, Any],
    shell_mesh: dict[str, Any],
    candidate_surfaces: list[SurfacePatch],
) -> None:
    shell_meshes = []
    if shell_mesh.get("faces"):
        shell_meshes.append(
            {
                "name": "envelope_shell",
                "vertices": shell_mesh.get("vertices", []),
                "faces": shell_mesh.get("faces", []),
            }
        )
    _write_text(job_dir / artifacts["shell_obj"], build_obj_text(shell_meshes))

    all_surface_meshes = [
        {
            "name": surface.surface_id,
            "vertices": surface.mesh["vertices"],
            "faces": surface.mesh["faces"],
        }
        for surface in candidate_surfaces
    ]
    _write_text(job_dir / artifacts["surfaces_all"], build_obj_text(all_surface_meshes))

    for surface_class, relative_path in artifacts["classes"].items():
        class_meshes = [
            {
                "name": surface.surface_id,
                "vertices": surface.mesh["vertices"],
                "faces": surface.mesh["faces"],
            }
            for surface in candidate_surfaces
            if surface.classification == surface_class
        ]
        _write_text(job_dir / relative_path, build_obj_text(class_meshes))


def _serialize_surface(surface: SurfacePatch) -> dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "object_name": surface.object_name,
        "space_global_id": surface.space_global_id,
        "space_express_id": surface.space_express_id,
        "space_name": surface.space_name,
        "classification": surface.classification,
        "area_m2": round(float(surface.area_m2), 6),
        "normal": _round_vector(surface.normal),
        "plane_normal": _round_vector(surface.normal),
        "plane_point": _round_vector(surface.plane_point),
        "centroid": _round_vector(surface.centroid),
        "polygon_components_3d": _serialize_polygon_components(surface),
        "triangles": [[_round_vector(vertex) for vertex in triangle.tolist()] for triangle in surface.triangles_3d],
        "source_surface_id": surface.source_surface_id,
        "subtracted_internal_surface_ids": list(surface.subtracted_internal_surface_ids or []),
        "reason": surface.reason,
        "artifacts": surface.artifacts or {},
    }


def _serialize_polygon_components(surface: SurfacePatch) -> list[list[list[list[float]]]]:
    components: list[list[list[list[float]]]] = []
    for polygon in _extract_polygons(surface.union_polygon_2d):
        component_rings = [
            [_round_vector(_lift_point(coordinate, surface.plane_point, surface.basis_u, surface.basis_v)) for coordinate in list(polygon.exterior.coords)[:-1]]
        ]
        for interior in polygon.interiors:
            component_rings.append(
                [_round_vector(_lift_point(coordinate, surface.plane_point, surface.basis_u, surface.basis_v)) for coordinate in list(interior.coords)[:-1]]
            )
        components.append(component_rings)
    return components


def _build_box_mesh(aabb_min: np.ndarray, aabb_max: np.ndarray) -> dict[str, Any]:
    min_x, min_y, min_z = (float(value) for value in aabb_min.tolist())
    max_x, max_y, max_z = (float(value) for value in aabb_max.tolist())
    vertices = [
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
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
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return {"vertices": vertices, "faces": faces}


def _build_face_adjacency(faces: np.ndarray) -> dict[int, list[int]]:
    adjacency = {index: [] for index in range(len(faces))}
    edge_map: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        edges = (
            tuple(sorted((int(face[0]), int(face[1])))),
            tuple(sorted((int(face[1]), int(face[2])))),
            tuple(sorted((int(face[2]), int(face[0])))),
        )
        for edge in edges:
            edge_map.setdefault(edge, []).append(face_index)
    for references in edge_map.values():
        if len(references) < 2:
            continue
        for left_index in references:
            for right_index in references:
                if left_index == right_index or right_index in adjacency[left_index]:
                    continue
                adjacency[left_index].append(right_index)
    return adjacency


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, reference))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = np.cross(normal, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v


def _project_triangle(
    triangle: np.ndarray,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray:
    relative = triangle - plane_point
    return np.stack((relative @ basis_u, relative @ basis_v), axis=1)


def _triangle_polygon(points_2d: np.ndarray) -> Polygon | None:
    polygon = Polygon([(float(x), float(y)) for x, y in points_2d])
    if polygon.is_empty or polygon.area <= FACE_AREA_EPSILON:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= FACE_AREA_EPSILON:
        return None
    if isinstance(polygon, MultiPolygon):
        polygons = [candidate for candidate in polygon.geoms if candidate.area > FACE_AREA_EPSILON]
        if not polygons:
            return None
        polygon = max(polygons, key=lambda candidate: candidate.area)
    return polygon if isinstance(polygon, Polygon) else None


def _extract_polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > FACE_AREA_EPSILON else []
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if polygon.area > FACE_AREA_EPSILON]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for child in geometry.geoms:
            polygons.extend(_extract_polygons(child))
        return polygons
    return []


def _triangulate_polygons_3d(
    polygons: list[Polygon],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[list[list[float]]]:
    triangles_3d: list[list[list[float]]] = []
    for polygon in polygons:
        for triangle in triangulate(polygon):
            if triangle.is_empty or triangle.area <= FACE_AREA_EPSILON:
                continue
            if not polygon.covers(triangle.representative_point()):
                continue
            coordinates = list(triangle.exterior.coords)[:-1]
            if len(coordinates) != 3:
                continue
            triangles_3d.append(
                [_lift_point(coordinate, plane_point, basis_u, basis_v) for coordinate in coordinates]
            )
    return triangles_3d


def _lift_point(
    coordinate_2d: tuple[float, float] | list[float],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[float]:
    x, y = coordinate_2d
    lifted = plane_point + (float(x) * basis_u) + (float(y) * basis_v)
    return [float(value) for value in lifted.tolist()]


def _triangles_to_mesh(triangles_3d: list[list[list[float]]]) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    for triangle in triangles_3d:
        face_indices: list[int] = []
        for vertex in triangle:
            rounded_vertex = tuple(round(float(component), 9) for component in vertex)
            index = vertex_index.get(rounded_vertex)
            if index is None:
                index = len(vertices)
                vertex_index[rounded_vertex] = index
                vertices.append([float(component) for component in vertex])
            face_indices.append(index)
        faces.append(face_indices)

    return {"vertices": vertices, "faces": faces}


def _entity_sort_key(entity: dict[str, Any]) -> tuple[str, int]:
    return (entity.get("global_id") or f"entity_{entity['express_id']}", entity["express_id"])


def _round_vector(values: np.ndarray | list[float]) -> list[float]:
    iterable = values.tolist() if isinstance(values, np.ndarray) else values
    return [round(float(value), 6) for value in iterable]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
