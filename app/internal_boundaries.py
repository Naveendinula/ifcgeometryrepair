from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as orient_polygon
from shapely.ops import triangulate

from .external_shell import (
    FACE_AREA_EPSILON,
    SurfacePatch,
    _extract_polygons,
    _extract_surface_patches_from_mesh,
    _lift_point,
    _plane_basis,
    _project_triangle,
    _triangle_polygon,
    _triangles_to_mesh,
)
from .mesh_normalizer import build_obj_text
from .polygon_clipper import BACKEND_NAME as CLIP_BACKEND_NAME, intersection as clip_intersection


PLANE_KEY_PRECISION = 6
OPPOSITE_NORMAL_EPSILON = 1e-3
SLIVER_AREA_THRESHOLD_M2 = 0.01


@dataclass(slots=True)
class InternalBoundaryResult:
    payload: dict[str, Any]


@dataclass(slots=True)
class PlanarPolygon:
    polygon_id: str
    space_global_id: str | None
    space_express_id: int
    space_name: str | None
    source_surface_id: str
    normal: np.ndarray
    plane_point: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    polygon_2d: Polygon


@dataclass(slots=True)
class SpaceGeometry:
    global_id: str | None
    express_id: int
    name: str | None
    object_name: str
    polygons: list[PlanarPolygon]
    aabb_min: np.ndarray
    aabb_max: np.ndarray


@dataclass(slots=True)
class IntersectionProjectionResult:
    left_polygons: list[Polygon]
    right_polygons: list[Polygon]
    shared_polygons: list[Polygon]
    shared_normal: np.ndarray
    shared_plane_point: np.ndarray
    shared_basis_u: np.ndarray
    shared_basis_v: np.ndarray
    raw_left_polygons: list[Polygon] = field(default_factory=list)
    raw_right_polygons: list[Polygon] = field(default_factory=list)
    raw_shared_polygons: list[Polygon] = field(default_factory=list)
    rejection_code: str | None = None
    rejection_message: str | None = None


def run_internal_boundary_generation(
    job_id: str,
    job_dir: Path,
    preprocessing_result: dict[str, Any],
    *,
    threshold_m: float,
) -> InternalBoundaryResult:
    geometry_dir = job_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    all_spaces = sorted(
        [entity for entity in preprocessing_result.get("entities", []) if entity.get("entity_type") == "IfcSpace"],
        key=_entity_sort_key,
    )

    processed_spaces: list[SpaceGeometry] = []
    skipped_spaces: list[dict[str, Any]] = []
    for space_index, entity in enumerate(all_spaces):
        built_space = _build_space_geometry(entity, space_index)
        if built_space is None:
            skipped_spaces.append(
                {
                    "global_id": entity.get("global_id"),
                    "express_id": entity["express_id"],
                    "name": entity.get("name"),
                    "reason": entity.get("reason") or "Space mesh unavailable",
                }
            )
            continue
        processed_spaces.append(built_space)

    candidate_pairs = _generate_candidate_pairs(processed_spaces, threshold_m)
    oriented_surfaces: list[dict[str, Any]] = []
    shared_surfaces: list[dict[str, Any]] = []
    rejected_shared_components: list[dict[str, Any]] = []
    adjacencies: list[dict[str, Any]] = []
    obj_meshes: list[dict[str, Any]] = []

    for pair_index, (space_a, space_b) in enumerate(candidate_pairs):
        pair_result = intersection_projection_sets(space_a, space_b, threshold_m, pair_index)
        rejected_shared_components.extend(pair_result.get("rejected_shared_components", []))
        if pair_result["oriented_surfaces"]:
            adjacencies.append(pair_result["adjacency"])
        oriented_surfaces.extend(pair_result["oriented_surfaces"])
        shared_surfaces.extend(pair_result["shared_surfaces"])
        obj_meshes.extend(pair_result["shared_meshes"])

    artifacts = {
        "detail": "geometry/internal_boundaries.json",
        "obj": "geometry/internal_boundaries.obj",
    }
    payload = {
        "job_id": job_id,
        "threshold_m": float(threshold_m),
        "epsilon": float(OPPOSITE_NORMAL_EPSILON),
        "clip_backend": CLIP_BACKEND_NAME,
        "summary": {
            "space_count": len(all_spaces),
            "processed_space_count": len(processed_spaces),
            "skipped_space_count": len(skipped_spaces),
            "candidate_pair_count": len(candidate_pairs),
            "adjacent_pair_count": len(adjacencies),
            "oriented_surface_count": len(oriented_surfaces),
            "shared_surface_count": len(shared_surfaces),
            "rejected_shared_component_count": len(rejected_shared_components),
            "total_oriented_area_m2": float(sum(surface["area_m2"] for surface in oriented_surfaces)),
            "total_shared_area_m2": float(sum(surface["area_m2"] for surface in shared_surfaces)),
        },
        "adjacencies": adjacencies,
        "oriented_surfaces": oriented_surfaces,
        "shared_surfaces": shared_surfaces,
        "rejected_shared_components": rejected_shared_components,
        "skipped_spaces": skipped_spaces,
        "artifacts": artifacts,
    }

    _write_json(geometry_dir / "internal_boundaries.json", payload)
    _write_text(geometry_dir / "internal_boundaries.obj", build_obj_text(obj_meshes))
    return InternalBoundaryResult(payload=payload)


def intersection_projection(
    left: PlanarPolygon,
    right: PlanarPolygon,
    threshold_m: float,
    *,
    epsilon: float = OPPOSITE_NORMAL_EPSILON,
) -> IntersectionProjectionResult:
    if float(np.dot(left.normal, right.normal)) >= -epsilon:
        return _empty_projection_result(
            left,
            rejection_code="non_opposite_normals",
            rejection_message=(
                f'Polygons "{left.polygon_id}" and "{right.polygon_id}" are not approximately opposite within epsilon.'
            ),
        )

    if _polygon_exceeds_plane_threshold(left, right.plane_point, right.normal, threshold_m):
        return _empty_projection_result(
            left,
            rejection_code="pair_distance_exceeds_threshold",
            rejection_message=(
                f'Polygon "{left.polygon_id}" exceeds the opposite-plane distance threshold against "{right.polygon_id}".'
            ),
        )
    if _polygon_exceeds_plane_threshold(right, left.plane_point, left.normal, threshold_m):
        return _empty_projection_result(
            left,
            rejection_code="pair_distance_exceeds_threshold",
            rejection_message=(
                f'Polygon "{right.polygon_id}" exceeds the opposite-plane distance threshold against "{left.polygon_id}".'
            ),
        )

    projected_left_on_right = _project_polygon_to_plane(
        left,
        right.plane_point,
        right.normal,
        right.basis_u,
        right.basis_v,
    )
    projected_right_on_left = _project_polygon_to_plane(
        right,
        left.plane_point,
        left.normal,
        left.basis_u,
        left.basis_v,
    )

    midpoint_normal, midpoint_point, midpoint_basis_u, midpoint_basis_v = _midpoint_plane(left, right)
    projected_left_on_midpoint = _project_polygon_to_plane(
        left,
        midpoint_point,
        midpoint_normal,
        midpoint_basis_u,
        midpoint_basis_v,
    )
    projected_right_on_midpoint = _project_polygon_to_plane(
        right,
        midpoint_point,
        midpoint_normal,
        midpoint_basis_u,
        midpoint_basis_v,
    )

    raw_left_polygons = clip_intersection(projected_left_on_right, right.polygon_2d)
    raw_right_polygons = clip_intersection(projected_right_on_left, left.polygon_2d)
    raw_shared_polygons = clip_intersection(projected_left_on_midpoint, projected_right_on_midpoint)

    left_polygons = _normalize_projected_results(
        raw_left_polygons,
        source_normal=left.normal,
        plane_normal=right.normal,
    )
    right_polygons = _normalize_projected_results(
        raw_right_polygons,
        source_normal=right.normal,
        plane_normal=left.normal,
    )
    shared_polygons = _normalize_projected_results(
        raw_shared_polygons,
        source_normal=midpoint_normal,
        plane_normal=midpoint_normal,
    )

    return IntersectionProjectionResult(
        left_polygons=left_polygons,
        right_polygons=right_polygons,
        shared_polygons=shared_polygons,
        shared_normal=midpoint_normal,
        shared_plane_point=midpoint_point,
        shared_basis_u=midpoint_basis_u,
        shared_basis_v=midpoint_basis_v,
        raw_left_polygons=raw_left_polygons,
        raw_right_polygons=raw_right_polygons,
        raw_shared_polygons=raw_shared_polygons,
    )


def intersection_projection_sets(
    space_a: SpaceGeometry,
    space_b: SpaceGeometry,
    threshold_m: float,
    pair_index: int,
) -> dict[str, Any]:
    oriented_surfaces: list[dict[str, Any]] = []
    shared_surfaces: list[dict[str, Any]] = []
    shared_meshes: list[dict[str, Any]] = []
    rejected_shared_components: list[dict[str, Any]] = []
    adjacency_oriented_ids: list[str] = []
    adjacency_shared_ids: list[str] = []
    adjacency_rejected_shared_ids: list[str] = []
    surface_index = 0

    for polygon_a in space_a.polygons:
        for polygon_b in space_b.polygons:
            result = intersection_projection(polygon_a, polygon_b, threshold_m)
            if (
                not result.left_polygons
                and not result.right_polygons
                and not result.shared_polygons
                and not result.raw_left_polygons
                and not result.raw_right_polygons
                and not result.raw_shared_polygons
            ):
                continue

            shared_components = _build_projection_components(
                result.raw_shared_polygons,
                result.shared_plane_point,
                result.shared_basis_u,
                result.shared_basis_v,
                result.shared_normal,
                result.shared_normal,
                result.shared_plane_point,
                result.shared_basis_u,
                result.shared_basis_v,
            )
            left_components = _build_projection_components(
                result.raw_left_polygons,
                right_plane_point := polygon_b.plane_point,
                right_basis_u := polygon_b.basis_u,
                right_basis_v := polygon_b.basis_v,
                polygon_a.normal,
                polygon_b.normal,
                result.shared_plane_point,
                result.shared_basis_u,
                result.shared_basis_v,
            )
            right_components = _build_projection_components(
                result.raw_right_polygons,
                left_plane_point := polygon_a.plane_point,
                left_basis_u := polygon_a.basis_u,
                left_basis_v := polygon_a.basis_v,
                polygon_b.normal,
                polygon_a.normal,
                result.shared_plane_point,
                result.shared_basis_u,
                result.shared_basis_v,
            )

            group_count = max(len(shared_components), len(left_components), len(right_components))
            for group_index in range(group_count):
                shared_component = shared_components[group_index] if group_index < len(shared_components) else None
                left_component = left_components[group_index] if group_index < len(left_components) else None
                right_component = right_components[group_index] if group_index < len(right_components) else None

                shared_polygon = shared_component["polygon"] if shared_component is not None else None
                left_polygon = left_component["polygon"] if left_component is not None else None
                right_polygon = right_component["polygon"] if right_component is not None else None

                shared_surface_id = f"ib_{pair_index}_{surface_index}"
                left_surface_id = f"ibo_{pair_index}_{surface_index}_a" if left_polygon is not None else None
                right_surface_id = f"ibo_{pair_index}_{surface_index}_b" if right_polygon is not None else None

                if left_polygon is not None and left_surface_id is not None:
                    left_surface = _build_oriented_surface_payload(
                        surface_id=left_surface_id,
                        polygon=left_polygon,
                        owner=polygon_a,
                        adjacent=polygon_b,
                        plane_point=right_plane_point,
                        basis_u=right_basis_u,
                        basis_v=right_basis_v,
                        plane_normal=polygon_a.normal,
                        reference_plane_normal=polygon_b.normal,
                        paired_surface_id=right_surface_id,
                        shared_surface_id=shared_surface_id,
                    )
                    oriented_surfaces.append(left_surface)
                    adjacency_oriented_ids.append(left_surface_id)

                if right_polygon is not None and right_surface_id is not None:
                    right_surface = _build_oriented_surface_payload(
                        surface_id=right_surface_id,
                        polygon=right_polygon,
                        owner=polygon_b,
                        adjacent=polygon_a,
                        plane_point=left_plane_point,
                        basis_u=left_basis_u,
                        basis_v=left_basis_v,
                        plane_normal=polygon_b.normal,
                        reference_plane_normal=polygon_a.normal,
                        paired_surface_id=left_surface_id,
                        shared_surface_id=shared_surface_id,
                    )
                    oriented_surfaces.append(right_surface)
                    adjacency_oriented_ids.append(right_surface_id)

                rejection = _build_rejected_shared_component(
                    pair_index=pair_index,
                    group_index=group_index,
                    polygon_a=polygon_a,
                    polygon_b=polygon_b,
                    shared_surface_id=shared_surface_id,
                    left_surface_id=left_surface_id,
                    right_surface_id=right_surface_id,
                    left_component=left_component,
                    right_component=right_component,
                    shared_component=shared_component,
                    left_component_count=len(left_components),
                    right_component_count=len(right_components),
                    shared_component_count=len(shared_components),
                )
                if rejection is not None:
                    rejected_shared_components.append(rejection)
                    adjacency_rejected_shared_ids.append(shared_surface_id)
                elif shared_polygon is not None:
                    shared_surface, shared_mesh = _build_shared_surface_payload(
                        surface_id=shared_surface_id,
                        polygon=shared_polygon,
                        space_a=polygon_a,
                        space_b=polygon_b,
                        plane_point=result.shared_plane_point,
                        basis_u=result.shared_basis_u,
                        basis_v=result.shared_basis_v,
                        plane_normal=result.shared_normal,
                        oriented_surface_ids=[surface_id for surface_id in (left_surface_id, right_surface_id) if surface_id],
                    )
                    shared_surfaces.append(shared_surface)
                    shared_meshes.append(shared_mesh)
                    adjacency_shared_ids.append(shared_surface_id)

                surface_index += 1

    if not oriented_surfaces:
        return {
            "adjacency": {},
            "oriented_surfaces": [],
            "shared_surfaces": [],
            "shared_meshes": [],
            "rejected_shared_components": rejected_shared_components,
        }

    adjacency = {
        "space_a_global_id": space_a.global_id,
        "space_a_express_id": space_a.express_id,
        "space_b_global_id": space_b.global_id,
        "space_b_express_id": space_b.express_id,
        "oriented_surface_ids": adjacency_oriented_ids,
        "shared_surface_ids": adjacency_shared_ids,
        "rejected_shared_surface_ids": adjacency_rejected_shared_ids,
        "rejected_shared_component_count": len(rejected_shared_components),
        "shared_area_m2": float(sum(surface["area_m2"] for surface in shared_surfaces)),
    }
    return {
        "adjacency": adjacency,
        "oriented_surfaces": oriented_surfaces,
        "shared_surfaces": shared_surfaces,
        "shared_meshes": shared_meshes,
        "rejected_shared_components": rejected_shared_components,
    }


def _build_space_geometry(entity: dict[str, Any], space_index: int) -> SpaceGeometry | None:
    mesh = entity.get("mesh")
    if not entity.get("valid") or not mesh or not mesh.get("vertices") or not mesh.get("faces"):
        return None

    vertices = np.asarray(mesh["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh["faces"], dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return None

    surface_patches = _extract_surface_patches_from_mesh(
        mesh,
        id_prefix=f"space_{space_index}",
        space_global_id=entity.get("global_id"),
        space_express_id=entity["express_id"],
        space_name=entity.get("name"),
    )
    polygons: list[PlanarPolygon] = []
    for patch in surface_patches:
        polygons.extend(_patch_to_planar_polygons(patch))
    if not polygons:
        return None

    return SpaceGeometry(
        global_id=entity.get("global_id"),
        express_id=entity["express_id"],
        name=entity.get("name"),
        object_name=entity.get("object_name") or entity.get("global_id") or f"entity_{entity['express_id']}",
        polygons=polygons,
        aabb_min=np.min(vertices, axis=0),
        aabb_max=np.max(vertices, axis=0),
    )


def _patch_to_planar_polygons(patch: SurfacePatch) -> list[PlanarPolygon]:
    planar_polygons: list[PlanarPolygon] = []
    for polygon_index, polygon in enumerate(_extract_polygons(patch.union_polygon_2d)):
        oriented = orient_polygon(polygon, sign=1.0)
        if oriented.is_empty or float(oriented.area) <= FACE_AREA_EPSILON:
            continue
        planar_polygons.append(
            PlanarPolygon(
                polygon_id=f"{patch.surface_id}_poly_{polygon_index}",
                space_global_id=patch.space_global_id,
                space_express_id=patch.space_express_id or -1,
                space_name=patch.space_name,
                source_surface_id=patch.surface_id,
                normal=np.asarray(patch.normal, dtype=np.float64),
                plane_point=np.asarray(patch.plane_point, dtype=np.float64),
                basis_u=np.asarray(patch.basis_u, dtype=np.float64),
                basis_v=np.asarray(patch.basis_v, dtype=np.float64),
                polygon_2d=oriented,
            )
        )
    return planar_polygons


def _generate_candidate_pairs(spaces: list[SpaceGeometry], threshold_m: float) -> list[tuple[SpaceGeometry, SpaceGeometry]]:
    if len(spaces) < 2:
        return []

    sorted_spaces = sorted(spaces, key=lambda space: (float(space.aabb_min[0]), *_space_sort_key(space)))
    active: list[SpaceGeometry] = []
    pair_map: dict[tuple[tuple[str, int], tuple[str, int]], tuple[SpaceGeometry, SpaceGeometry]] = {}

    for current in sorted_spaces:
        active = [space for space in active if float(space.aabb_max[0]) + threshold_m >= float(current.aabb_min[0])]
        for candidate in active:
            if _aabb_gap(candidate, current) > threshold_m:
                continue
            ordered_pair = tuple(sorted((candidate, current), key=_space_sort_key))
            pair_map[(_space_sort_key(ordered_pair[0]), _space_sort_key(ordered_pair[1]))] = ordered_pair
        active.append(current)

    return [pair_map[key] for key in sorted(pair_map)]


def _polygon_exceeds_plane_threshold(
    polygon: PlanarPolygon,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    threshold_m: float,
) -> bool:
    for point in _polygon_ring_points_3d(polygon.polygon_2d, polygon.plane_point, polygon.basis_u, polygon.basis_v):
        distance = abs(float(np.dot(plane_normal, point - plane_point)))
        if distance > threshold_m:
            return True
    return False


def _project_polygon_to_plane(
    polygon: PlanarPolygon,
    target_plane_point: np.ndarray,
    target_plane_normal: np.ndarray,
    target_basis_u: np.ndarray,
    target_basis_v: np.ndarray,
) -> Polygon:
    shell = _project_ring_to_plane(
        list(polygon.polygon_2d.exterior.coords)[:-1],
        polygon.plane_point,
        polygon.basis_u,
        polygon.basis_v,
        target_plane_point,
        target_plane_normal,
        target_basis_u,
        target_basis_v,
    )
    holes = [
        _project_ring_to_plane(
            list(interior.coords)[:-1],
            polygon.plane_point,
            polygon.basis_u,
            polygon.basis_v,
            target_plane_point,
            target_plane_normal,
            target_basis_u,
            target_basis_v,
        )
        for interior in polygon.polygon_2d.interiors
    ]
    projected = Polygon(shell, holes)
    if not projected.is_valid:
        projected = projected.buffer(0)
    if projected.is_empty:
        return Polygon()
    if isinstance(projected, Polygon):
        return projected
    extracted = _extract_polygons(projected)
    return extracted[0] if len(extracted) == 1 else projected


def _project_ring_to_plane(
    ring: list[tuple[float, float]],
    source_plane_point: np.ndarray,
    source_basis_u: np.ndarray,
    source_basis_v: np.ndarray,
    target_plane_point: np.ndarray,
    target_plane_normal: np.ndarray,
    target_basis_u: np.ndarray,
    target_basis_v: np.ndarray,
) -> list[tuple[float, float]]:
    projected_ring: list[tuple[float, float]] = []
    for coordinate in ring:
        point_3d = np.asarray(_lift_point(coordinate, source_plane_point, source_basis_u, source_basis_v), dtype=np.float64)
        projected_point = point_3d - (np.dot(target_plane_normal, point_3d - target_plane_point) * target_plane_normal)
        relative = projected_point - target_plane_point
        projected_ring.append((float(relative @ target_basis_u), float(relative @ target_basis_v)))
    return projected_ring


def _normalize_projected_results(
    polygons: list[Polygon],
    *,
    source_normal: np.ndarray,
    plane_normal: np.ndarray,
) -> list[Polygon]:
    exterior_ccw = float(np.dot(source_normal, plane_normal)) >= 0.0
    normalized: list[Polygon] = []
    for polygon in polygons:
        if polygon.is_empty or float(polygon.area) < SLIVER_AREA_THRESHOLD_M2:
            continue
        oriented = orient_polygon(polygon, sign=1.0 if exterior_ccw else -1.0)
        if oriented.is_empty or float(oriented.area) < SLIVER_AREA_THRESHOLD_M2:
            continue
        normalized.append(oriented)
    return normalized


def _midpoint_plane(
    left: PlanarPolygon,
    right: PlanarPolygon,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = left.normal / np.linalg.norm(left.normal)
    left_offset = float(np.dot(normal, left.plane_point))
    right_offset = float(np.dot(normal, right.plane_point))
    midpoint_offset = (left_offset + right_offset) / 2.0
    plane_point = normal * midpoint_offset
    basis_u, basis_v = _plane_basis(normal)
    return normal, plane_point, basis_u, basis_v


def _sort_polygons_by_reference(
    polygons: list[Polygon],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    reference_plane_point: np.ndarray,
    reference_basis_u: np.ndarray,
    reference_basis_v: np.ndarray,
) -> list[Polygon]:
    def sort_key(polygon: Polygon) -> tuple[float, float, float]:
        centroid_3d = np.asarray(_lift_point(polygon.centroid.coords[0], plane_point, basis_u, basis_v), dtype=np.float64)
        relative = centroid_3d - reference_plane_point
        return (
            round(float(relative @ reference_basis_u), PLANE_KEY_PRECISION),
            round(float(relative @ reference_basis_v), PLANE_KEY_PRECISION),
            round(float(centroid_3d[2]), PLANE_KEY_PRECISION),
        )

    return sorted(polygons, key=sort_key)


def _build_projection_components(
    polygons: list[Polygon],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    source_normal: np.ndarray,
    plane_normal: np.ndarray,
    reference_plane_point: np.ndarray,
    reference_basis_u: np.ndarray,
    reference_basis_v: np.ndarray,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for polygon in _sort_polygons_by_reference(
        polygons,
        plane_point,
        basis_u,
        basis_v,
        reference_plane_point,
        reference_basis_u,
        reference_basis_v,
    ):
        components.append(
            {
                "raw_polygon": polygon,
                "polygon": _normalize_projected_polygon(
                    polygon,
                    source_normal=source_normal,
                    plane_normal=plane_normal,
                ),
                "raw_area_m2": float(polygon.area),
            }
        )
    return components


def _normalize_projected_polygon(
    polygon: Polygon,
    *,
    source_normal: np.ndarray,
    plane_normal: np.ndarray,
) -> Polygon | None:
    if polygon.is_empty or float(polygon.area) < SLIVER_AREA_THRESHOLD_M2:
        return None
    exterior_ccw = float(np.dot(source_normal, plane_normal)) >= 0.0
    oriented = orient_polygon(polygon, sign=1.0 if exterior_ccw else -1.0)
    if oriented.is_empty or float(oriented.area) < SLIVER_AREA_THRESHOLD_M2:
        return None
    return oriented


def _build_rejected_shared_component(
    *,
    pair_index: int,
    group_index: int,
    polygon_a: PlanarPolygon,
    polygon_b: PlanarPolygon,
    shared_surface_id: str,
    left_surface_id: str | None,
    right_surface_id: str | None,
    left_component: dict[str, Any] | None,
    right_component: dict[str, Any] | None,
    shared_component: dict[str, Any] | None,
    left_component_count: int,
    right_component_count: int,
    shared_component_count: int,
) -> dict[str, Any] | None:
    counts_mismatch = len({left_component_count, right_component_count, shared_component_count}) > 1
    missing_component = left_component is None or right_component is None or shared_component is None
    rejected_left = left_component is not None and left_component.get("polygon") is None
    rejected_right = right_component is not None and right_component.get("polygon") is None
    rejected_shared = shared_component is not None and shared_component.get("polygon") is None

    rejection_code: str | None = None
    rejection_message: str | None = None
    if counts_mismatch and missing_component:
        rejection_code = "component_count_mismatch"
        rejection_message = (
            f'Midpoint component group {group_index} for spaces "{polygon_a.space_global_id or polygon_a.space_express_id}" '
            f'and "{polygon_b.space_global_id or polygon_b.space_express_id}" does not map one-to-one after sorting '
            f"(left={left_component_count}, right={right_component_count}, shared={shared_component_count})."
        )
    elif shared_component is None:
        rejection_code = "missing_shared_overlap"
        rejection_message = (
            f'Midpoint component group {group_index} for polygons "{polygon_a.polygon_id}" and "{polygon_b.polygon_id}" '
            "does not have a valid midpoint overlap."
        )
    elif rejected_shared:
        rejection_code = "degenerate_midpoint_overlap"
        rejection_message = (
            f'Midpoint component group {group_index} for polygons "{polygon_a.polygon_id}" and "{polygon_b.polygon_id}" '
            "collapsed below the shared sliver threshold."
        )
    elif left_component is None or right_component is None:
        rejection_code = "missing_projected_overlap"
        rejection_message = (
            f'Midpoint component group {group_index} for polygons "{polygon_a.polygon_id}" and "{polygon_b.polygon_id}" '
            "is missing one side of the clipped overlap set."
        )
    elif rejected_left or rejected_right:
        rejection_code = "degenerate_projected_overlap"
        rejection_message = (
            f'Midpoint component group {group_index} for polygons "{polygon_a.polygon_id}" and "{polygon_b.polygon_id}" '
            "collapsed below the projected-overlap sliver threshold."
        )

    if rejection_code is None:
        return None

    return {
        "pair_index": pair_index,
        "group_index": group_index,
        "shared_surface_id": shared_surface_id,
        "space_a_global_id": polygon_a.space_global_id,
        "space_a_express_id": polygon_a.space_express_id,
        "space_b_global_id": polygon_b.space_global_id,
        "space_b_express_id": polygon_b.space_express_id,
        "source_surface_a_id": polygon_a.source_surface_id,
        "source_surface_b_id": polygon_b.source_surface_id,
        "source_polygon_a_id": polygon_a.polygon_id,
        "source_polygon_b_id": polygon_b.polygon_id,
        "oriented_surface_ids": [surface_id for surface_id in (left_surface_id, right_surface_id) if surface_id],
        "left_area_m2": _component_area(left_component),
        "right_area_m2": _component_area(right_component),
        "shared_area_m2": _component_area(shared_component),
        "rejection_code": rejection_code,
        "rejection_message": rejection_message,
    }


def _component_area(component: dict[str, Any] | None) -> float | None:
    if component is None:
        return None
    raw_area = component.get("raw_area_m2")
    if raw_area is None:
        return None
    return float(raw_area)


def _build_oriented_surface_payload(
    *,
    surface_id: str,
    polygon: Polygon,
    owner: PlanarPolygon,
    adjacent: PlanarPolygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    plane_normal: np.ndarray,
    reference_plane_normal: np.ndarray,
    paired_surface_id: str | None,
    shared_surface_id: str | None,
) -> dict[str, Any]:
    exterior_ccw = float(np.dot(plane_normal, reference_plane_normal)) >= 0.0
    triangles_3d = _triangulate_polygon_3d(
        polygon,
        plane_point,
        basis_u,
        basis_v,
        exterior_ccw=exterior_ccw,
    )
    centroid_3d = _lift_point(polygon.centroid.coords[0], plane_point, basis_u, basis_v)
    polygon_vertices_3d = [_lift_point(coordinate, plane_point, basis_u, basis_v) for coordinate in list(polygon.exterior.coords)[:-1]]
    polygon_rings_3d = _lift_polygon_rings(polygon, plane_point, basis_u, basis_v)

    return {
        "oriented_surface_id": surface_id,
        "object_name": surface_id,
        "space_global_id": owner.space_global_id,
        "space_express_id": owner.space_express_id,
        "space_name": owner.space_name,
        "adjacent_space_global_id": adjacent.space_global_id,
        "adjacent_space_express_id": adjacent.space_express_id,
        "adjacent_space_name": adjacent.space_name,
        "source_surface_id": owner.source_surface_id,
        "source_polygon_id": owner.polygon_id,
        "paired_surface_id": paired_surface_id,
        "shared_surface_id": shared_surface_id,
        "area_m2": float(polygon.area),
        "plane_normal": _round_vector(plane_normal),
        "plane_point": _round_vector(plane_point),
        "centroid": _round_vector(centroid_3d),
        "polygon_vertices_3d": [_round_vector(vertex) for vertex in polygon_vertices_3d],
        "polygon_rings_3d": [[_round_vector(vertex) for vertex in ring] for ring in polygon_rings_3d],
        "triangles": [[_round_vector(vertex) for vertex in triangle] for triangle in triangles_3d],
    }


def _build_shared_surface_payload(
    *,
    surface_id: str,
    polygon: Polygon,
    space_a: PlanarPolygon,
    space_b: PlanarPolygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    plane_normal: np.ndarray,
    oriented_surface_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    triangles_3d = _triangulate_polygon_3d(
        polygon,
        plane_point,
        basis_u,
        basis_v,
        exterior_ccw=True,
    )
    centroid_3d = _lift_point(polygon.centroid.coords[0], plane_point, basis_u, basis_v)
    polygon_vertices_3d = [_lift_point(coordinate, plane_point, basis_u, basis_v) for coordinate in list(polygon.exterior.coords)[:-1]]
    polygon_rings_3d = _lift_polygon_rings(polygon, plane_point, basis_u, basis_v)
    mesh = _triangles_to_mesh(triangles_3d)

    payload = {
        "shared_surface_id": surface_id,
        "object_name": surface_id,
        "space_a_global_id": space_a.space_global_id,
        "space_a_express_id": space_a.space_express_id,
        "space_b_global_id": space_b.space_global_id,
        "space_b_express_id": space_b.space_express_id,
        "oriented_surface_ids": oriented_surface_ids,
        "area_m2": float(polygon.area),
        "plane_normal": _round_vector(plane_normal),
        "plane_point": _round_vector(plane_point),
        "centroid": _round_vector(centroid_3d),
        "polygon_vertices_3d": [_round_vector(vertex) for vertex in polygon_vertices_3d],
        "polygon_rings_3d": [[_round_vector(vertex) for vertex in ring] for ring in polygon_rings_3d],
        "triangles": [[_round_vector(vertex) for vertex in triangle] for triangle in triangles_3d],
    }
    mesh_payload = {
        "name": surface_id,
        "vertices": mesh["vertices"],
        "faces": mesh["faces"],
    }
    return payload, mesh_payload


def _triangulate_polygon_3d(
    polygon: Polygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    *,
    exterior_ccw: bool,
) -> list[list[list[float]]]:
    triangles_3d: list[list[list[float]]] = []
    for triangle in triangulate(polygon):
        if triangle.is_empty or triangle.area <= FACE_AREA_EPSILON:
            continue
        if not polygon.covers(triangle.representative_point()):
            continue
        oriented_triangle = orient_polygon(triangle, sign=1.0 if exterior_ccw else -1.0)
        coordinates = list(oriented_triangle.exterior.coords)[:-1]
        if len(coordinates) != 3:
            continue
        triangles_3d.append(
            [
                _lift_point(coordinate, plane_point, basis_u, basis_v)
                for coordinate in coordinates
            ]
        )
    return triangles_3d


def _lift_polygon_rings(
    polygon: Polygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[list[list[float]]]:
    rings = [
        [_lift_point(coordinate, plane_point, basis_u, basis_v) for coordinate in list(polygon.exterior.coords)[:-1]]
    ]
    for interior in polygon.interiors:
        rings.append(
            [_lift_point(coordinate, plane_point, basis_u, basis_v) for coordinate in list(interior.coords)[:-1]]
        )
    return rings


def _polygon_ring_points_3d(
    polygon: Polygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for ring in [polygon.exterior, *polygon.interiors]:
        for coordinate in list(ring.coords)[:-1]:
            points.append(np.asarray(_lift_point(coordinate, plane_point, basis_u, basis_v), dtype=np.float64))
    return points


def _empty_projection_result(
    reference: PlanarPolygon,
    *,
    rejection_code: str | None = None,
    rejection_message: str | None = None,
) -> IntersectionProjectionResult:
    return IntersectionProjectionResult(
        left_polygons=[],
        right_polygons=[],
        shared_polygons=[],
        shared_normal=reference.normal,
        shared_plane_point=reference.plane_point,
        shared_basis_u=reference.basis_u,
        shared_basis_v=reference.basis_v,
        raw_left_polygons=[],
        raw_right_polygons=[],
        raw_shared_polygons=[],
        rejection_code=rejection_code,
        rejection_message=rejection_message,
    )


def _aabb_gap(left: SpaceGeometry, right: SpaceGeometry) -> float:
    deltas = []
    for axis in range(3):
        if left.aabb_max[axis] < right.aabb_min[axis]:
            deltas.append(float(right.aabb_min[axis] - left.aabb_max[axis]))
        elif right.aabb_max[axis] < left.aabb_min[axis]:
            deltas.append(float(left.aabb_min[axis] - right.aabb_max[axis]))
        else:
            deltas.append(0.0)
    return float(np.linalg.norm(np.asarray(deltas, dtype=np.float64)))


def _entity_sort_key(entity: dict[str, Any]) -> tuple[str, int]:
    return (entity.get("global_id") or f"entity_{entity['express_id']}", entity["express_id"])


def _space_sort_key(space: SpaceGeometry) -> tuple[str, int]:
    return (space.global_id or f"entity_{space.express_id}", space.express_id)


def _round_vector(values: np.ndarray | list[float]) -> list[float]:
    if isinstance(values, np.ndarray):
        iterable = values.tolist()
    else:
        iterable = values
    return [round(float(value), 6) for value in iterable]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
