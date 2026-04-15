from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .internal_boundaries import (
    OPPOSITE_NORMAL_EPSILON,
    PlanarPolygon,
    SpaceGeometry,
    _build_space_geometry,
    _entity_sort_key,
    _extract_polygons,
    _lift_point,
    _lift_polygon_rings,
    _plane_basis,
    _round_vector,
    _sort_polygons_by_reference,
    _triangulate_polygon_3d,
    _triangles_to_mesh,
)
from .mesh_normalizer import build_obj_text
from .polygon_clipper import BACKEND_NAME as CLIP_BACKEND_NAME, difference as clip_difference, union as clip_union


FACE_AREA_EPSILON = 1e-9
PLANE_OFFSET_TOLERANCE_M = 1e-3


@dataclass(slots=True)
class ExternalCandidatesResult:
    payload: dict[str, Any]


@dataclass(slots=True)
class InternalSurfaceRef:
    surface_id: str
    space_global_id: str | None
    space_express_id: int | None
    source_surface_id: str | None
    normal: np.ndarray
    plane_point: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    polygon_2d: Polygon | MultiPolygon


def run_external_candidate_generation(
    job_id: str,
    job_dir: Path,
    preprocessing_result: dict[str, Any],
    internal_boundary_result: dict[str, Any],
) -> ExternalCandidatesResult:
    external_candidates_dir = job_dir / "geometry" / "external_candidates"
    external_candidates_dir.mkdir(parents=True, exist_ok=True)

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

    internal_surfaces_by_space = _build_internal_surfaces_by_space(internal_boundary_result)
    candidate_surfaces: list[dict[str, Any]] = []
    candidate_meshes: list[dict[str, Any]] = []
    spaces_payload: list[dict[str, Any]] = []
    source_polygon_count = 0
    subtracted_source_polygon_count = 0
    space_candidate_id_map: dict[tuple[str | None, int], list[str]] = {}

    for space_index, space in enumerate(processed_spaces):
        internal_refs = internal_surfaces_by_space.get(_space_key(space.global_id, space.express_id), [])
        local_candidates, local_meshes, local_source_count, local_subtracted_count = _subtract_internal_surfaces(
            space,
            internal_refs,
            space_index=space_index,
        )
        candidate_surfaces.extend(local_candidates)
        candidate_meshes.extend(local_meshes)
        source_polygon_count += local_source_count
        subtracted_source_polygon_count += local_subtracted_count
        space_candidate_id_map[_space_key(space.global_id, space.express_id)] = [
            surface["surface_id"] for surface in local_candidates
        ]

    spaces_payload = [
        {
            "global_id": space.global_id,
            "express_id": space.express_id,
            "name": space.name,
            "candidate_surface_ids": space_candidate_id_map.get(_space_key(space.global_id, space.express_id), []),
        }
        for space in processed_spaces
    ]

    artifacts = {
        "result": "geometry/external_candidates/result.json",
        "candidates_all": "geometry/external_candidates/candidates_all.obj",
    }
    payload = {
        "job_id": job_id,
        "epsilon": float(OPPOSITE_NORMAL_EPSILON),
        "plane_offset_tolerance_m": float(PLANE_OFFSET_TOLERANCE_M),
        "clip_backend": CLIP_BACKEND_NAME,
        "summary": {
            "space_count": len(all_spaces),
            "processed_space_count": len(processed_spaces),
            "skipped_space_count": len(skipped_spaces),
            "source_polygon_count": source_polygon_count,
            "subtracted_source_polygon_count": subtracted_source_polygon_count,
            "candidate_surface_count": len(candidate_surfaces),
            "total_candidate_area_m2": float(sum(surface["area_m2"] for surface in candidate_surfaces)),
        },
        "spaces": spaces_payload,
        "candidate_surfaces": candidate_surfaces,
        "skipped_spaces": skipped_spaces,
        "artifacts": artifacts,
    }

    _write_json(external_candidates_dir / "result.json", payload)
    _write_text(external_candidates_dir / "candidates_all.obj", build_obj_text(candidate_meshes))
    return ExternalCandidatesResult(payload=payload)


def _subtract_internal_surfaces(
    space: SpaceGeometry,
    internal_refs: list[InternalSurfaceRef],
    *,
    space_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    candidate_surfaces: list[dict[str, Any]] = []
    candidate_meshes: list[dict[str, Any]] = []
    candidate_index = 0
    subtracted_source_polygon_count = 0

    for polygon in space.polygons:
        ordered_polygons: list[Polygon] = [polygon.polygon_2d]
        subtractor_refs = _collect_coplanar_subtractors(polygon, internal_refs)
        if subtractor_refs:
            subtractor_geometries = [reference.polygon_2d for reference in subtractor_refs]
            subtractor_union = _geometry_from_polygons(clip_union(subtractor_geometries))
            if subtractor_union is not None:
                difference_polygons = clip_difference(polygon.polygon_2d, subtractor_union)
                ordered_polygons = _extract_polygons(_geometry_from_polygons(difference_polygons))
            was_subtracted = not (
                len(ordered_polygons) == 1 and ordered_polygons[0].equals(polygon.polygon_2d)
            )
            if was_subtracted:
                subtracted_source_polygon_count += 1

        ordered_polygons = _sort_polygons_by_reference(
            ordered_polygons,
            polygon.plane_point,
            polygon.basis_u,
            polygon.basis_v,
            polygon.plane_point,
            polygon.basis_u,
            polygon.basis_v,
        )
        for remainder_index, remainder in enumerate(ordered_polygons):
            if remainder.is_empty or float(remainder.area) <= FACE_AREA_EPSILON:
                continue
            surface_id = f"ec_{space_index}_{candidate_index}"
            payload, mesh = _build_candidate_surface_payload(
                surface_id=surface_id,
                polygon=remainder,
                owner=polygon,
                subtracted_internal_surface_ids=[reference.surface_id for reference in subtractor_refs],
            )
            payload["object_name"] = surface_id
            payload["fragment_index"] = remainder_index
            candidate_surfaces.append(payload)
            candidate_meshes.append(mesh)
            candidate_index += 1

    return candidate_surfaces, candidate_meshes, len(space.polygons), subtracted_source_polygon_count


def _build_internal_surfaces_by_space(
    internal_boundary_result: dict[str, Any],
) -> dict[tuple[str | None, int], list[InternalSurfaceRef]]:
    refs_by_space: dict[tuple[str | None, int], list[InternalSurfaceRef]] = {}
    surfaces_by_id = {
        surface["oriented_surface_id"]: surface
        for surface in internal_boundary_result.get("oriented_surfaces", [])
        if surface.get("oriented_surface_id")
    }
    for surface in internal_boundary_result.get("oriented_surfaces", []):
        space_express_id = surface.get("space_express_id")
        if not isinstance(space_express_id, int):
            continue
        paired_surface = surfaces_by_id.get(surface.get("paired_surface_id"))
        ref = _oriented_surface_to_ref(surface, paired_surface)
        if ref is None:
            continue
        refs_by_space.setdefault(_space_key(ref.space_global_id, ref.space_express_id), []).append(ref)
    return refs_by_space


def _oriented_surface_to_ref(
    surface: dict[str, Any],
    paired_surface: dict[str, Any] | None,
) -> InternalSurfaceRef | None:
    normal = np.asarray(surface.get("plane_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= FACE_AREA_EPSILON:
        return None
    normal /= magnitude
    geometry_surface = paired_surface or surface
    plane_point = np.asarray(geometry_surface.get("plane_point", [0.0, 0.0, 0.0]), dtype=np.float64)
    basis_u, basis_v = _plane_basis(normal)
    polygon_2d = _polygon_from_rings_3d(geometry_surface.get("polygon_rings_3d", []), plane_point, basis_u, basis_v)
    if polygon_2d is None or polygon_2d.is_empty or float(polygon_2d.area) <= FACE_AREA_EPSILON:
        return None
    return InternalSurfaceRef(
        surface_id=surface["oriented_surface_id"],
        space_global_id=surface.get("space_global_id"),
        space_express_id=surface.get("space_express_id"),
        source_surface_id=surface.get("source_surface_id"),
        normal=normal,
        plane_point=plane_point,
        basis_u=basis_u,
        basis_v=basis_v,
        polygon_2d=polygon_2d,
    )


def _collect_coplanar_subtractors(
    source_polygon: PlanarPolygon,
    internal_refs: list[InternalSurfaceRef],
) -> list[InternalSurfaceRef]:
    subtractors: list[InternalSurfaceRef] = []
    for reference in internal_refs:
        if not _are_coplanar(source_polygon, reference):
            continue
        projected_polygon = _reproject_polygon_to_source_basis(reference, source_polygon)
        if projected_polygon is None:
            continue
        overlap_area = float(source_polygon.polygon_2d.intersection(projected_polygon).area)
        if overlap_area <= FACE_AREA_EPSILON:
            continue
        subtractors.append(
            InternalSurfaceRef(
                surface_id=reference.surface_id,
                space_global_id=reference.space_global_id,
                space_express_id=reference.space_express_id,
                source_surface_id=reference.source_surface_id,
                normal=reference.normal,
                plane_point=source_polygon.plane_point,
                basis_u=source_polygon.basis_u,
                basis_v=source_polygon.basis_v,
                polygon_2d=projected_polygon,
            )
        )
    return subtractors


def _are_coplanar(source_polygon: PlanarPolygon, reference: InternalSurfaceRef) -> bool:
    normal_alignment = abs(abs(float(np.dot(source_polygon.normal, reference.normal))) - 1.0)
    if normal_alignment > OPPOSITE_NORMAL_EPSILON:
        return False

    source_offset = _plane_offset(source_polygon.normal, source_polygon.plane_point)
    reference_offset = _plane_offset(reference.normal, reference.plane_point)
    if float(np.dot(source_polygon.normal, reference.normal)) >= 0.0:
        return abs(source_offset - reference_offset) <= PLANE_OFFSET_TOLERANCE_M
    return abs(source_offset + reference_offset) <= PLANE_OFFSET_TOLERANCE_M


def _build_candidate_surface_payload(
    *,
    surface_id: str,
    polygon: Polygon,
    owner: PlanarPolygon,
    subtracted_internal_surface_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    triangles_3d = _triangulate_polygon_3d(
        polygon,
        owner.plane_point,
        owner.basis_u,
        owner.basis_v,
        exterior_ccw=True,
    )
    centroid_3d = _lift_point(polygon.centroid.coords[0], owner.plane_point, owner.basis_u, owner.basis_v)
    polygon_vertices_3d = [
        _lift_point(coordinate, owner.plane_point, owner.basis_u, owner.basis_v)
        for coordinate in list(polygon.exterior.coords)[:-1]
    ]
    polygon_rings_3d = _lift_polygon_rings(polygon, owner.plane_point, owner.basis_u, owner.basis_v)
    mesh = _triangles_to_mesh(triangles_3d)

    payload = {
        "surface_id": surface_id,
        "space_global_id": owner.space_global_id,
        "space_express_id": owner.space_express_id,
        "space_name": owner.space_name,
        "source_surface_id": owner.source_surface_id,
        "source_polygon_id": owner.polygon_id,
        "area_m2": float(polygon.area),
        "plane_normal": _round_vector(owner.normal),
        "plane_point": _round_vector(owner.plane_point),
        "centroid": _round_vector(centroid_3d),
        "polygon_vertices_3d": [_round_vector(vertex) for vertex in polygon_vertices_3d],
        "polygon_rings_3d": [[_round_vector(vertex) for vertex in ring] for ring in polygon_rings_3d],
        "triangles": [[_round_vector(vertex) for vertex in triangle] for triangle in triangles_3d],
        "subtracted_internal_surface_ids": subtracted_internal_surface_ids,
    }
    mesh_payload = {
        "name": surface_id,
        "vertices": mesh["vertices"],
        "faces": mesh["faces"],
    }
    return payload, mesh_payload


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
    extracted = _extract_polygons(polygon)
    if not extracted:
        return None
    if len(extracted) == 1:
        return extracted[0]
    return unary_union(extracted)


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


def _reproject_polygon_to_source_basis(
    reference: InternalSurfaceRef,
    source_polygon: PlanarPolygon,
) -> Polygon | MultiPolygon | None:
    reprojected_polygons: list[Polygon] = []
    for polygon in _extract_polygons(reference.polygon_2d):
        shell = _reproject_ring(
            list(polygon.exterior.coords)[:-1],
            reference.plane_point,
            reference.basis_u,
            reference.basis_v,
            source_polygon.plane_point,
            source_polygon.basis_u,
            source_polygon.basis_v,
        )
        holes = [
            _reproject_ring(
                list(interior.coords)[:-1],
                reference.plane_point,
                reference.basis_u,
                reference.basis_v,
                source_polygon.plane_point,
                source_polygon.basis_u,
                source_polygon.basis_v,
            )
            for interior in polygon.interiors
        ]
        projected_polygon = Polygon(shell, holes)
        if not projected_polygon.is_valid:
            projected_polygon = projected_polygon.buffer(0)
        reprojected_polygons.extend(_extract_polygons(projected_polygon))

    if not reprojected_polygons:
        return None
    if len(reprojected_polygons) == 1:
        return reprojected_polygons[0]
    return unary_union(reprojected_polygons)


def _reproject_ring(
    ring: list[tuple[float, float]],
    source_plane_point: np.ndarray,
    source_basis_u: np.ndarray,
    source_basis_v: np.ndarray,
    target_plane_point: np.ndarray,
    target_basis_u: np.ndarray,
    target_basis_v: np.ndarray,
) -> list[tuple[float, float]]:
    projected: list[tuple[float, float]] = []
    for x, y in ring:
        point_3d = source_plane_point + (float(x) * source_basis_u) + (float(y) * source_basis_v)
        relative = point_3d - target_plane_point
        projected.append((float(relative @ target_basis_u), float(relative @ target_basis_v)))
    return projected


def _plane_offset(normal: np.ndarray, plane_point: np.ndarray) -> float:
    return float(np.dot(normal, plane_point))


def _space_key(global_id: str | None, express_id: int) -> tuple[str | None, int]:
    return global_id, express_id


def _geometry_from_polygons(polygons: list[Polygon] | Polygon | MultiPolygon | None) -> Polygon | MultiPolygon | None:
    if polygons is None:
        return None
    if isinstance(polygons, (Polygon, MultiPolygon)):
        return polygons
    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return unary_union(polygons)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
