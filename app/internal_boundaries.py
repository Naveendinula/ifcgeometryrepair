from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union

from .mesh_normalizer import build_obj_text


FACE_AREA_EPSILON = 1e-9
PLANE_KEY_PRECISION = 6
ANTIPARALLEL_DEGREES = 5.0
ANTIPARALLEL_DOT_THRESHOLD = -float(np.cos(np.deg2rad(ANTIPARALLEL_DEGREES)))
SLIVER_AREA_THRESHOLD_M2 = 0.01


@dataclass(slots=True)
class InternalBoundaryResult:
    payload: dict[str, Any]


@dataclass(slots=True)
class FaceGeometry:
    triangle: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    area: float


@dataclass(slots=True)
class SpaceGeometry:
    global_id: str | None
    express_id: int
    name: str | None
    object_name: str
    vertices: np.ndarray
    faces: np.ndarray
    face_geometries: list[FaceGeometry]
    aabb_min: np.ndarray
    aabb_max: np.ndarray


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
    for entity in all_spaces:
        built_space = _build_space_geometry(entity)
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
    shared_surfaces: list[dict[str, Any]] = []
    adjacencies: list[dict[str, Any]] = []
    obj_meshes: list[dict[str, Any]] = []

    for pair_index, (space_a, space_b) in enumerate(candidate_pairs):
        pair_surfaces = _detect_pair_shared_surfaces(space_a, space_b, threshold_m, pair_index)
        if not pair_surfaces:
            continue

        shared_surface_ids = [surface["shared_surface_id"] for surface in pair_surfaces]
        shared_area_m2 = float(sum(surface["area_m2"] for surface in pair_surfaces))
        adjacencies.append(
            {
                "space_a_global_id": space_a.global_id,
                "space_b_global_id": space_b.global_id,
                "shared_surface_ids": shared_surface_ids,
                "shared_area_m2": shared_area_m2,
            }
        )
        shared_surfaces.extend(pair_surfaces)
        obj_meshes.extend(
            {
                "name": surface["shared_surface_id"],
                "vertices": surface["mesh"]["vertices"],
                "faces": surface["mesh"]["faces"],
            }
            for surface in pair_surfaces
        )

    artifacts = {
        "detail": "geometry/internal_boundaries.json",
        "obj": "geometry/internal_boundaries.obj",
    }
    payload = {
        "job_id": job_id,
        "threshold_m": float(threshold_m),
        "summary": {
            "space_count": len(all_spaces),
            "processed_space_count": len(processed_spaces),
            "skipped_space_count": len(skipped_spaces),
            "candidate_pair_count": len(candidate_pairs),
            "adjacent_pair_count": len(adjacencies),
            "shared_surface_count": len(shared_surfaces),
            "total_shared_area_m2": float(sum(surface["area_m2"] for surface in shared_surfaces)),
        },
        "adjacencies": adjacencies,
        "shared_surfaces": [
            {
                key: value
                for key, value in surface.items()
                if key != "mesh"
            }
            for surface in shared_surfaces
        ],
        "skipped_spaces": skipped_spaces,
        "artifacts": artifacts,
    }

    _write_json(geometry_dir / "internal_boundaries.json", payload)
    _write_text(geometry_dir / "internal_boundaries.obj", build_obj_text(obj_meshes))
    return InternalBoundaryResult(payload=payload)


def _build_space_geometry(entity: dict[str, Any]) -> SpaceGeometry | None:
    mesh = entity.get("mesh")
    if not entity.get("valid") or not mesh or not mesh.get("vertices") or not mesh.get("faces"):
        return None

    vertices = np.asarray(mesh["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh["faces"], dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return None

    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    magnitudes = np.linalg.norm(cross_products, axis=1)
    valid_mask = magnitudes > FACE_AREA_EPSILON
    if not np.any(valid_mask):
        return None

    triangles = triangles[valid_mask]
    magnitudes = magnitudes[valid_mask]
    normals = cross_products[valid_mask] / magnitudes[:, np.newaxis]
    centroids = np.mean(triangles, axis=1)
    areas = 0.5 * magnitudes
    face_geometries = [
        FaceGeometry(
            triangle=triangle,
            centroid=centroid,
            normal=normal,
            area=float(area),
        )
        for triangle, centroid, normal, area in zip(triangles, centroids, normals, areas, strict=False)
    ]

    return SpaceGeometry(
        global_id=entity.get("global_id"),
        express_id=entity["express_id"],
        name=entity.get("name"),
        object_name=entity.get("object_name") or entity.get("global_id") or f"entity_{entity['express_id']}",
        vertices=vertices,
        faces=faces,
        face_geometries=face_geometries,
        aabb_min=np.min(vertices, axis=0),
        aabb_max=np.max(vertices, axis=0),
    )


def _generate_candidate_pairs(spaces: list[SpaceGeometry], threshold_m: float) -> list[tuple[SpaceGeometry, SpaceGeometry]]:
    if len(spaces) < 2:
        return []

    sorted_spaces = sorted(spaces, key=lambda space: (float(space.aabb_min[0]), *_space_sort_key(space)))
    active: list[SpaceGeometry] = []
    pair_map: dict[tuple[tuple[str | None, int], tuple[str | None, int]], tuple[SpaceGeometry, SpaceGeometry]] = {}

    for current in sorted_spaces:
        active = [space for space in active if float(space.aabb_max[0]) + threshold_m >= float(current.aabb_min[0])]
        for candidate in active:
            if _aabb_gap(candidate, current) > threshold_m:
                continue
            ordered_pair = tuple(sorted((candidate, current), key=_space_sort_key))
            pair_map[(_space_sort_key(ordered_pair[0]), _space_sort_key(ordered_pair[1]))] = ordered_pair
        active.append(current)

    return [pair_map[key] for key in sorted(pair_map)]


def _detect_pair_shared_surfaces(
    space_a: SpaceGeometry,
    space_b: SpaceGeometry,
    threshold_m: float,
    pair_index: int,
) -> list[dict[str, Any]]:
    plane_buckets: dict[tuple[Any, ...], dict[str, Any]] = {}

    for face_a in space_a.face_geometries:
        for face_b in space_b.face_geometries:
            if np.dot(face_a.normal, face_b.normal) > ANTIPARALLEL_DOT_THRESHOLD:
                continue

            plane_offset_a = float(np.dot(face_a.normal, face_a.centroid))
            plane_offset_b = float(np.dot(face_a.normal, face_b.centroid))
            plane_separation = abs(plane_offset_b - plane_offset_a)
            if plane_separation > threshold_m:
                continue

            midpoint_offset = (plane_offset_a + plane_offset_b) / 2.0
            plane_key = _plane_bucket_key(space_a, space_b, face_a.normal, midpoint_offset)
            bucket = plane_buckets.get(plane_key)
            if bucket is None:
                basis_u, basis_v = _plane_basis(face_a.normal)
                plane_origin = face_a.normal * midpoint_offset
                bucket = {
                    "normal": face_a.normal,
                    "plane_point": plane_origin,
                    "basis_u": basis_u,
                    "basis_v": basis_v,
                    "polygons": [],
                }
                plane_buckets[plane_key] = bucket

            triangle_a_2d = _project_triangle(face_a.triangle, bucket["plane_point"], bucket["basis_u"], bucket["basis_v"])
            triangle_b_2d = _project_triangle(face_b.triangle, bucket["plane_point"], bucket["basis_u"], bucket["basis_v"])
            polygon_a = _triangle_polygon(triangle_a_2d)
            polygon_b = _triangle_polygon(triangle_b_2d)
            if polygon_a is None or polygon_b is None:
                continue

            intersection = polygon_a.intersection(polygon_b)
            bucket["polygons"].extend(_extract_polygons(intersection))

    pair_surfaces: list[dict[str, Any]] = []
    surface_index = 0

    for plane_key in sorted(plane_buckets):
        bucket = plane_buckets[plane_key]
        if not bucket["polygons"]:
            continue

        unioned = unary_union(bucket["polygons"])
        for polygon in _extract_polygons(unioned):
            area_m2 = float(polygon.area)
            if area_m2 < SLIVER_AREA_THRESHOLD_M2:
                continue

            triangles_3d = _triangulate_polygon_3d(polygon, bucket["plane_point"], bucket["basis_u"], bucket["basis_v"])
            if not triangles_3d:
                continue

            mesh = _triangles_to_mesh(triangles_3d)
            polygon_vertices_3d = [
                _lift_point(coordinate, bucket["plane_point"], bucket["basis_u"], bucket["basis_v"])
                for coordinate in list(polygon.exterior.coords)[:-1]
            ]
            centroid_3d = _lift_point(polygon.centroid.coords[0], bucket["plane_point"], bucket["basis_u"], bucket["basis_v"])
            shared_surface_id = f"ib_{pair_index}_{surface_index}"
            pair_surfaces.append(
                {
                    "shared_surface_id": shared_surface_id,
                    "space_a_global_id": space_a.global_id,
                    "space_b_global_id": space_b.global_id,
                    "area_m2": area_m2,
                    "plane_normal": _round_vector(bucket["normal"]),
                    "plane_point": _round_vector(bucket["plane_point"]),
                    "centroid": _round_vector(centroid_3d),
                    "polygon_vertices_3d": [_round_vector(vertex) for vertex in polygon_vertices_3d],
                    "triangles": [[_round_vector(vertex) for vertex in triangle] for triangle in triangles_3d],
                    "mesh": mesh,
                }
            )
            surface_index += 1

    return pair_surfaces


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
    return np.stack(
        (
            relative @ basis_u,
            relative @ basis_v,
        ),
        axis=1,
    )


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


def _triangulate_polygon_3d(
    polygon: Polygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[list[list[float]]]:
    triangles_3d: list[list[list[float]]] = []
    for triangle in triangulate(polygon):
        if triangle.is_empty or triangle.area <= FACE_AREA_EPSILON:
            continue
        if not polygon.covers(triangle.representative_point()):
            continue
        coordinates = list(triangle.exterior.coords)[:-1]
        if len(coordinates) != 3:
            continue
        triangles_3d.append(
            [
                _lift_point(coordinate, plane_point, basis_u, basis_v)
                for coordinate in coordinates
            ]
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

    return {
        "vertices": vertices,
        "faces": faces,
    }


def _plane_bucket_key(
    space_a: SpaceGeometry,
    space_b: SpaceGeometry,
    normal: np.ndarray,
    midpoint_offset: float,
) -> tuple[Any, ...]:
    return (
        _space_sort_key(space_a),
        _space_sort_key(space_b),
        tuple(round(float(component), PLANE_KEY_PRECISION) for component in normal.tolist()),
        round(float(midpoint_offset), PLANE_KEY_PRECISION),
    )


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
