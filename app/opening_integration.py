from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as orient_polygon

from .external_shell import (
    FACE_AREA_EPSILON,
    SurfacePatch,
    _extract_polygons,
    _extract_surface_patches_from_mesh,
    _lift_point,
    _plane_basis,
    _triangles_to_mesh,
)
from .mesh_normalizer import build_obj_text
from .polygon_clipper import (
    BACKEND_NAME as CLIP_BACKEND_NAME,
    difference as clip_difference,
    intersection as clip_intersection,
)

OPPOSITE_NORMAL_EPSILON = 1e-3
PLANE_OFFSET_TOLERANCE_M = 1e-3
NORMAL_CLUSTER_DOT_THRESHOLD = 1.0 - 1e-3


@dataclass(slots=True)
class OpeningIntegrationResult:
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# maxA — select the largest face cluster aligned with a target normal
# ---------------------------------------------------------------------------


def max_a(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_normal: np.ndarray,
    epsilon: float = OPPOSITE_NORMAL_EPSILON,
) -> dict[str, Any] | None:
    """Return the largest coplanar face cluster whose normal matches *target_normal*.

    Returns a dict with keys ``normal``, ``plane_point``, ``basis_u``, ``basis_v``,
    ``polygon_2d`` (Shapely Polygon), and ``area_m2``, or *None* when no cluster
    matches.
    """
    if len(vertices) == 0 or len(faces) == 0:
        return None

    target_normal = np.asarray(target_normal, dtype=np.float64)
    target_len = np.linalg.norm(target_normal)
    if target_len < 1e-12:
        return None
    target_normal = target_normal / target_len

    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    magnitudes = np.linalg.norm(cross_products, axis=1)
    valid = magnitudes > FACE_AREA_EPSILON
    if not np.any(valid):
        return None

    face_normals = cross_products[valid] / magnitudes[valid, np.newaxis]
    face_areas = magnitudes[valid] * 0.5
    valid_faces = faces[valid]
    valid_triangles = triangles[valid]

    # Keep only faces whose normal is within epsilon of the target.
    dots = face_normals @ target_normal
    aligned = dots >= (1.0 - epsilon)
    if not np.any(aligned):
        return None

    aligned_normals = face_normals[aligned]
    aligned_areas = face_areas[aligned]
    aligned_triangles = valid_triangles[aligned]

    # Group aligned faces into coplanar clusters.
    centroids = np.mean(aligned_triangles, axis=1)
    avg_normal = np.mean(aligned_normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)
    plane_offsets = centroids @ avg_normal

    clusters: list[list[int]] = []
    assigned = np.zeros(len(aligned_areas), dtype=bool)
    for i in range(len(aligned_areas)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(aligned_areas)):
            if assigned[j]:
                continue
            if abs(float(plane_offsets[j] - plane_offsets[i])) <= PLANE_OFFSET_TOLERANCE_M:
                if float(aligned_normals[i] @ aligned_normals[j]) >= NORMAL_CLUSTER_DOT_THRESHOLD:
                    cluster.append(j)
                    assigned[j] = True
        clusters.append(cluster)

    # Pick the cluster with the largest total area.
    best_cluster: list[int] | None = None
    best_area = 0.0
    for cluster in clusters:
        cluster_area = float(np.sum(aligned_areas[cluster]))
        if cluster_area > best_area:
            best_area = cluster_area
            best_cluster = cluster
    if best_cluster is None:
        return None

    cluster_normals = aligned_normals[best_cluster]
    cluster_centroids = centroids[best_cluster]
    cluster_triangles = aligned_triangles[best_cluster]

    normal = np.mean(cluster_normals, axis=0)
    normal /= np.linalg.norm(normal)
    plane_point = normal * float(np.mean(cluster_centroids @ normal))
    basis_u, basis_v = _plane_basis(normal)

    # Build a 2-D Shapely polygon by uniting the projected triangles.
    from shapely.ops import unary_union

    polygons_2d: list[Polygon] = []
    for tri_3d in cluster_triangles:
        relative = tri_3d - plane_point
        coords_2d = [(float(r @ basis_u), float(r @ basis_v)) for r in relative]
        tri_poly = Polygon(coords_2d)
        if tri_poly.is_valid and tri_poly.area > FACE_AREA_EPSILON:
            polygons_2d.append(tri_poly)

    if not polygons_2d:
        return None

    merged = unary_union(polygons_2d)
    extracted = _extract_polygons(merged)
    if not extracted:
        return None

    # Take the largest polygon if union produced multiple disjoint parts.
    polygon_2d = max(extracted, key=lambda p: p.area)
    polygon_2d = orient_polygon(polygon_2d, sign=1.0)

    return {
        "normal": normal,
        "plane_point": plane_point,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "polygon_2d": polygon_2d,
        "area_m2": float(polygon_2d.area),
    }


# ---------------------------------------------------------------------------
# Project opening polygons onto boundary surfaces
# ---------------------------------------------------------------------------


def _is_coplanar(
    opening_normal: np.ndarray,
    opening_plane_point: np.ndarray,
    boundary_normal: np.ndarray,
    boundary_plane_point: np.ndarray,
    threshold_m: float,
    epsilon: float = OPPOSITE_NORMAL_EPSILON,
) -> bool:
    """Check whether an opening face and a boundary surface are approximately coplanar."""
    dot = abs(float(opening_normal @ boundary_normal))
    if dot < 1.0 - epsilon:
        return False
    # Check plane distance.
    plane_distance = abs(float(boundary_normal @ (opening_plane_point - boundary_plane_point)))
    return plane_distance <= threshold_m


def _project_polygon_to_basis(
    polygon_2d: Polygon,
    source_plane_point: np.ndarray,
    source_basis_u: np.ndarray,
    source_basis_v: np.ndarray,
    target_plane_point: np.ndarray,
    target_plane_normal: np.ndarray,
    target_basis_u: np.ndarray,
    target_basis_v: np.ndarray,
) -> Polygon:
    """Project a 2-D polygon from one plane basis to another."""
    def _project_ring(ring_coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        projected: list[tuple[float, float]] = []
        for u, v in ring_coords:
            point_3d = source_plane_point + u * source_basis_u + v * source_basis_v
            # Project onto target plane.
            offset = float(target_plane_normal @ (point_3d - target_plane_point))
            point_on_plane = point_3d - offset * target_plane_normal
            relative = point_on_plane - target_plane_point
            projected.append((float(relative @ target_basis_u), float(relative @ target_basis_v)))
        return projected

    shell = _project_ring(list(polygon_2d.exterior.coords)[:-1])
    holes = [_project_ring(list(interior.coords)[:-1]) for interior in polygon_2d.interiors]
    result = Polygon(shell, holes)
    if not result.is_valid:
        result = result.buffer(0)
    if result.is_empty:
        return Polygon()
    if isinstance(result, Polygon):
        return result
    extracted = _extract_polygons(result)
    return extracted[0] if extracted else Polygon()


def project_openings_onto_boundaries(
    opening_entities: list[dict[str, Any]],
    internal_surfaces: list[dict[str, Any]],
    external_surfaces: list[dict[str, Any]],
    threshold_m: float,
    min_area_m2: float,
    epsilon: float = OPPOSITE_NORMAL_EPSILON,
) -> dict[str, Any]:
    """Project opening faces onto internal and external boundary surfaces.

    For each boundary surface, the opening is intersected and the remainder is
    computed.  Returns a dict with ``opening_surfaces``, ``modified_boundaries``,
    and ``summary``.
    """
    opening_faces = _extract_opening_faces(opening_entities, threshold_m, epsilon)

    opening_surfaces: list[dict[str, Any]] = []
    modified_boundaries: list[dict[str, Any]] = []
    openings_matched = 0

    # Exclude unclassified surfaces from being opening parents.
    _EXCLUDED_OPENING_PARENT_CLASSIFICATIONS = frozenset({"unclassified"})

    all_boundaries = [
        *[
            _boundary_record_from_internal(surface)
            for surface in internal_surfaces
        ],
        *[
            _boundary_record_from_external(surface)
            for surface in external_surfaces
            if surface.get("classification", "unclassified") not in _EXCLUDED_OPENING_PARENT_CLASSIFICATIONS
        ],
    ]

    for boundary in all_boundaries:
        b_normal = np.asarray(boundary["normal"], dtype=np.float64)
        b_plane_point = np.asarray(boundary["plane_point"], dtype=np.float64)
        b_basis_u, b_basis_v = _plane_basis(b_normal)
        b_polygon = _reconstruct_polygon(boundary, b_plane_point, b_basis_u, b_basis_v)
        if b_polygon is None or b_polygon.is_empty:
            modified_boundaries.append(boundary)
            continue

        remainder = b_polygon
        boundary_opening_ids: list[str] = []

        for opening_face in opening_faces:
            o_normal = opening_face["normal"]
            o_plane_point = opening_face["plane_point"]

            if not _is_coplanar(o_normal, o_plane_point, b_normal, b_plane_point, threshold_m, epsilon):
                continue

            projected = _project_polygon_to_basis(
                opening_face["polygon_2d"],
                o_plane_point,
                opening_face["basis_u"],
                opening_face["basis_v"],
                b_plane_point,
                b_normal,
                b_basis_u,
                b_basis_v,
            )
            if projected.is_empty:
                continue

            clipped = clip_intersection(remainder, projected)
            if not clipped:
                continue

            for clipped_polygon in clipped:
                if clipped_polygon.area < min_area_m2:
                    continue
                openings_matched += 1
                surface_id = f"oi_{boundary['surface_id']}_{opening_face['opening_id']}"
                opening_surfaces.append(
                    _build_opening_surface_payload(
                        surface_id=surface_id,
                        polygon_2d=clipped_polygon,
                        normal=b_normal,
                        plane_point=b_plane_point,
                        basis_u=b_basis_u,
                        basis_v=b_basis_v,
                        boundary=boundary,
                        opening_face=opening_face,
                    )
                )
                boundary_opening_ids.append(surface_id)

            diff_result = clip_difference(remainder, projected)
            if diff_result:
                merged = diff_result[0]
                for p in diff_result[1:]:
                    merged = merged.union(p)
                remainder = merged
            else:
                remainder = Polygon()

        # Filter tiny remainder fragments.
        remainder_polygons = _extract_polygons(remainder) if not remainder.is_empty else []
        remainder_polygons = [p for p in remainder_polygons if p.area >= min_area_m2]

        modified = dict(boundary)
        modified["opening_surface_ids"] = boundary_opening_ids
        if remainder_polygons:
            combined = remainder_polygons[0]
            for p in remainder_polygons[1:]:
                combined = combined.union(p)
            modified["remainder_polygon_rings_3d"] = _polygon_to_rings_3d(combined, b_plane_point, b_basis_u, b_basis_v)
            modified["remainder_area_m2"] = float(combined.area)
        else:
            modified["remainder_polygon_rings_3d"] = []
            modified["remainder_area_m2"] = 0.0
        modified_boundaries.append(modified)

    # -----------------------------------------------------------------------
    # Resolve each opening to one authoritative parent boundary.
    # When the same opening face is projected onto multiple coplanar
    # boundaries (e.g. an oriented internal surface AND a shell-derived
    # external surface), keep only the best match per opening per normal
    # direction.  Priority is based on classification (not boundary_type,
    # because internal_void surfaces have boundary_type="external" which
    # would incorrectly beat shared internal surfaces).
    # -----------------------------------------------------------------------
    _PARENT_CLASSIFICATION_PRIORITY = {
        "external_wall": 0,
        "roof": 0,
        "ground_floor": 0,
        "internal": 1,
        "internal_void": 2,
        "unknown": 3,
        "unclassified": 3,
    }

    grouped: dict[tuple[int, tuple[float, ...]], list[dict[str, Any]]] = {}
    for surface in opening_surfaces:
        normal_key = tuple(round(float(v), 3) for v in surface.get("normal", [0, 0, 0]))
        key = (surface["opening_express_id"], normal_key)
        grouped.setdefault(key, []).append(surface)

    deduplicated_opening_surfaces: list[dict[str, Any]] = []
    deduplicated_ids: set[str] = set()
    for _key, candidates in grouped.items():
        best = min(
            candidates,
            key=lambda s: (
                _PARENT_CLASSIFICATION_PRIORITY.get(s.get("boundary_classification", "unknown"), 3),
                -s.get("area_m2", 0.0),
            ),
        )
        deduplicated_opening_surfaces.append(best)
        deduplicated_ids.add(best["surface_id"])

    # Remove non-authoritative opening references from modified_boundaries.
    for boundary in modified_boundaries:
        boundary["opening_surface_ids"] = [
            sid for sid in boundary.get("opening_surface_ids", [])
            if sid in deduplicated_ids
        ]

    opening_surfaces = deduplicated_opening_surfaces
    openings_matched = len(opening_surfaces)

    return {
        "opening_surfaces": opening_surfaces,
        "modified_boundaries": modified_boundaries,
        "summary": {
            "openings_processed": len(opening_faces),
            "opening_surfaces_created": len(opening_surfaces),
            "boundaries_with_openings": sum(
                1 for b in modified_boundaries if b.get("opening_surface_ids")
            ),
            "total_opening_area_m2": float(
                sum(s["area_m2"] for s in opening_surfaces)
            ),
        },
    }


# ---------------------------------------------------------------------------
# Helpers — extracting opening representative faces
# ---------------------------------------------------------------------------


def _extract_opening_faces(
    opening_entities: list[dict[str, Any]],
    threshold_m: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    """For each opening entity, extract surface patches and return all planar faces."""
    opening_faces: list[dict[str, Any]] = []
    for entity in opening_entities:
        mesh = entity.get("mesh")
        if not entity.get("valid") or not mesh:
            continue
        vertices = np.asarray(mesh.get("vertices", []), dtype=np.float64).reshape(-1, 3)
        faces_arr = np.asarray(mesh.get("faces", []), dtype=np.int64).reshape(-1, 3)
        if len(vertices) == 0 or len(faces_arr) == 0:
            continue

        patches = _extract_surface_patches_from_mesh(
            mesh,
            id_prefix=f"opening_{entity['express_id']}",
            space_global_id=entity.get("global_id"),
            space_express_id=entity["express_id"],
            space_name=entity.get("name"),
        )
        for patch in patches:
            # Use max_a to get the dominant face aligned with this patch's normal.
            result = max_a(vertices, faces_arr, patch.normal, epsilon)
            if result is None:
                continue
            opening_faces.append({
                "opening_id": f"opening_{entity['express_id']}",
                "opening_express_id": entity["express_id"],
                "opening_global_id": entity.get("global_id"),
                "opening_name": entity.get("name"),
                "patch_surface_id": patch.surface_id,
                **result,
            })
    # Deduplicate: keep only the largest face per opening per normal direction.
    deduped: dict[tuple[int, tuple[float, float, float]], dict[str, Any]] = {}
    for face in opening_faces:
        normal_key = tuple(round(float(v), 3) for v in face["normal"])
        key = (face["opening_express_id"], normal_key)
        if key not in deduped or face["area_m2"] > deduped[key]["area_m2"]:
            deduped[key] = face
    return list(deduped.values())


# ---------------------------------------------------------------------------
# Helpers — boundary record normalization
# ---------------------------------------------------------------------------


def _boundary_record_from_internal(surface: dict[str, Any]) -> dict[str, Any]:
    surface_id = surface.get("oriented_surface_id") or surface.get("shared_surface_id", "")
    return {
        "surface_id": surface_id,
        "boundary_type": "internal",
        "space_global_id": surface.get("space_global_id"),
        "space_express_id": surface.get("space_express_id"),
        "space_name": surface.get("space_name"),
        "normal": surface.get("plane_normal", [0, 0, 0]),
        "plane_point": surface.get("plane_point", [0, 0, 0]),
        "centroid": surface.get("centroid", [0, 0, 0]),
        "area_m2": surface.get("area_m2", 0.0),
        "polygon_rings_3d": surface.get("polygon_rings_3d", []),
        "classification": "internal",
    }


def _boundary_record_from_external(surface: dict[str, Any]) -> dict[str, Any]:
    surface_id = surface.get("surface_id", "")
    return {
        "surface_id": surface_id,
        "boundary_type": "external",
        "space_global_id": surface.get("space_global_id"),
        "space_express_id": surface.get("space_express_id"),
        "space_name": surface.get("space_name"),
        "normal": surface.get("normal", [0, 0, 0]),
        "plane_point": surface.get("plane_point", [0, 0, 0]),
        "centroid": surface.get("centroid", [0, 0, 0]),
        "area_m2": surface.get("area_m2", 0.0),
        "polygon_rings_3d": surface.get("polygon_rings_3d", []),
        "classification": surface.get("classification", "unclassified"),
    }


def _reconstruct_polygon(
    boundary: dict[str, Any],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> Polygon | None:
    """Reconstruct a 2-D Shapely polygon from 3-D ring coordinates stored in a boundary record."""
    rings_3d = boundary.get("polygon_rings_3d", [])
    if not rings_3d:
        return None

    def _ring_to_2d(ring_3d: list[list[float]]) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for point in ring_3d:
            relative = np.asarray(point, dtype=np.float64) - plane_point
            coords.append((float(relative @ basis_u), float(relative @ basis_v)))
        return coords

    shell = _ring_to_2d(rings_3d[0])
    holes = [_ring_to_2d(ring) for ring in rings_3d[1:]]
    result = Polygon(shell, holes)
    if not result.is_valid:
        result = result.buffer(0)
    if result.is_empty:
        return None
    if isinstance(result, Polygon):
        return result
    extracted = _extract_polygons(result)
    return extracted[0] if extracted else None


# ---------------------------------------------------------------------------
# Helpers — output payloads
# ---------------------------------------------------------------------------


def _polygon_to_rings_3d(
    polygon: Polygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[list[list[float]]]:
    # Handle MultiPolygon by collecting rings from all constituent polygons.
    parts = _extract_polygons(polygon) if not isinstance(polygon, Polygon) else [polygon]
    rings: list[list[list[float]]] = []
    for part in parts:
        rings.append([_lift_point(c, plane_point, basis_u, basis_v) for c in list(part.exterior.coords)[:-1]])
        for interior in part.interiors:
            rings.append([_lift_point(c, plane_point, basis_u, basis_v) for c in list(interior.coords)[:-1]])
    return rings


def _build_opening_surface_payload(
    *,
    surface_id: str,
    polygon_2d: Polygon,
    normal: np.ndarray,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    boundary: dict[str, Any],
    opening_face: dict[str, Any],
) -> dict[str, Any]:
    centroid_2d = polygon_2d.centroid.coords[0]
    centroid_3d = _lift_point(centroid_2d, plane_point, basis_u, basis_v)
    polygon_rings_3d = _polygon_to_rings_3d(polygon_2d, plane_point, basis_u, basis_v)

    from shapely.ops import triangulate as shapely_triangulate

    triangles_3d: list[list[list[float]]] = []
    for tri in shapely_triangulate(polygon_2d):
        if tri.is_empty or tri.area <= FACE_AREA_EPSILON:
            continue
        if not polygon_2d.covers(tri.representative_point()):
            continue
        coords = list(tri.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        triangles_3d.append([_lift_point(c, plane_point, basis_u, basis_v) for c in coords])

    mesh = _triangles_to_mesh(triangles_3d)

    return {
        "surface_id": surface_id,
        "object_name": surface_id,
        "boundary_type": boundary.get("boundary_type", "unknown"),
        "boundary_surface_id": boundary["surface_id"],
        "boundary_classification": boundary.get("classification", "unclassified"),
        "opening_express_id": opening_face["opening_express_id"],
        "opening_global_id": opening_face.get("opening_global_id"),
        "opening_name": opening_face.get("opening_name"),
        "space_global_id": boundary.get("space_global_id"),
        "space_express_id": boundary.get("space_express_id"),
        "space_name": boundary.get("space_name"),
        "classification": "opening",
        "area_m2": round(float(polygon_2d.area), 6),
        "normal": _round_vector(normal),
        "plane_point": _round_vector(plane_point),
        "centroid": _round_vector(centroid_3d),
        "polygon_rings_3d": [[_round_vector(v) for v in ring] for ring in polygon_rings_3d],
        "triangles": [[_round_vector(v) for v in tri] for tri in triangles_3d],
        "mesh": mesh,
    }


def _round_vector(v: np.ndarray | list[float]) -> list[float]:
    if isinstance(v, np.ndarray):
        return [round(float(x), 6) for x in v.tolist()]
    return [round(float(x), 6) for x in v]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_opening_integration(
    job_id: str,
    job_dir: Path,
    preprocessing_result: dict[str, Any],
    internal_boundary_result: dict[str, Any],
    external_candidates_result: dict[str, Any],
    external_shell_result: dict[str, Any],
    *,
    threshold_m: float,
    min_area_m2: float,
) -> OpeningIntegrationResult:
    geometry_dir = job_dir / "geometry"
    integration_dir = geometry_dir / "opening_integration"
    integration_dir.mkdir(parents=True, exist_ok=True)

    opening_entities = [
        entity
        for entity in preprocessing_result.get("entities", [])
        if entity.get("entity_type") == "IfcOpeningElement" and entity.get("valid")
    ]

    internal_surfaces = internal_boundary_result.get("oriented_surfaces", [])
    # Merge external shell classification with external candidate geometry.
    external_surfaces = _merge_external_surfaces(
        external_shell_result.get("surfaces", []),
        external_candidates_result.get("candidate_surfaces", []),
    )

    if not opening_entities:
        payload = _empty_payload(job_id, threshold_m, min_area_m2)
        _write_json(geometry_dir / "opening_integration.json", payload)
        return OpeningIntegrationResult(payload=payload)

    result = project_openings_onto_boundaries(
        opening_entities,
        internal_surfaces,
        external_surfaces,
        threshold_m=threshold_m,
        min_area_m2=min_area_m2,
    )

    # Write debug OBJ for opening surfaces.
    obj_meshes: list[dict[str, Any]] = []
    for surface in result["opening_surfaces"]:
        mesh = surface.get("mesh")
        if mesh and mesh.get("vertices") and mesh.get("faces"):
            obj_meshes.append({
                "name": surface["surface_id"],
                "vertices": mesh["vertices"],
                "faces": mesh["faces"],
            })
    _write_text(integration_dir / "openings_projected.obj", build_obj_text(obj_meshes))

    artifacts = {
        "detail": "geometry/opening_integration.json",
        "obj": "geometry/opening_integration/openings_projected.obj",
    }

    payload = {
        "job_id": job_id,
        "threshold_m": float(threshold_m),
        "min_area_m2": float(min_area_m2),
        "clip_backend": CLIP_BACKEND_NAME,
        "summary": result["summary"],
        "opening_surfaces": result["opening_surfaces"],
        "modified_boundaries": [
            {k: v for k, v in b.items() if k != "mesh"}
            for b in result["modified_boundaries"]
        ],
        "artifacts": artifacts,
    }

    _write_json(geometry_dir / "opening_integration.json", payload)
    return OpeningIntegrationResult(payload=payload)


def _empty_payload(job_id: str, threshold_m: float, min_area_m2: float) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "threshold_m": float(threshold_m),
        "min_area_m2": float(min_area_m2),
        "clip_backend": CLIP_BACKEND_NAME,
        "summary": {
            "openings_processed": 0,
            "opening_surfaces_created": 0,
            "boundaries_with_openings": 0,
            "total_opening_area_m2": 0.0,
        },
        "opening_surfaces": [],
        "modified_boundaries": [],
        "artifacts": {},
    }


def _merge_external_surfaces(
    shell_surfaces: list[dict[str, Any]],
    candidate_surfaces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge classification from shell surfaces with geometry from candidate surfaces.

    The shell surfaces have ``classification`` but lack ``plane_point`` and
    ``polygon_rings_3d``.  The candidate surfaces have full geometry but no
    classification.  We match by ``surface_id`` / ``source_surface_id``.
    """
    candidate_by_id: dict[str, dict[str, Any]] = {
        s["surface_id"]: s for s in candidate_surfaces if s.get("surface_id")
    }
    merged: list[dict[str, Any]] = []
    for shell_surface in shell_surfaces:
        surface_id = shell_surface.get("surface_id", "")
        source_id = shell_surface.get("source_surface_id") or surface_id
        candidate = candidate_by_id.get(source_id) or candidate_by_id.get(surface_id) or {}
        record = dict(candidate) if candidate else dict(shell_surface)
        record["classification"] = shell_surface.get("classification", "unclassified")
        record["surface_id"] = surface_id
        # Ensure normal and plane_point are present (from candidate geometry).
        if "plane_normal" in record and "normal" not in record:
            record["normal"] = record["plane_normal"]
        elif "normal" not in record:
            record["normal"] = [0, 0, 0]
        if "plane_point" not in record:
            record["plane_point"] = record.get("centroid", [0, 0, 0])
        merged.append(record)
    return merged


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
