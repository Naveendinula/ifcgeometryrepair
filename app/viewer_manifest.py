from __future__ import annotations

import math
from typing import Any

DEFAULT_MIN_AREA_THRESHOLD_M2 = 0.25
DEFAULT_PROXIMITY_THRESHOLD_M = 0.05


def build_viewer_manifest(
    job_id: str,
    output_payload: dict[str, Any],
    *,
    internal_boundaries_payload: dict[str, Any] | None = None,
    min_area_threshold_m2: float = DEFAULT_MIN_AREA_THRESHOLD_M2,
    proximity_threshold_m: float = DEFAULT_PROXIMITY_THRESHOLD_M,
) -> dict[str, Any]:
    preprocessing = output_payload.get("preprocessing", {})
    preprocessing_artifacts = preprocessing.get("artifacts", {})
    preflight = output_payload.get("preflight", {})
    external_shell = output_payload.get("external_shell", {})
    external_shell_artifacts = external_shell.get("artifacts", {})
    opening_integration = output_payload.get("opening_integration", {})
    opening_integration_artifacts = opening_integration.get("artifacts", {})
    internal_boundaries = output_payload.get("internal_boundaries", {})
    internal_boundaries_artifacts = internal_boundaries.get("artifacts", {})

    entities = [*_build_entities(output_payload.get("spaces", [])), *_build_entities(output_payload.get("openings", []))]
    failed_entities = [entity for entity in entities if entity["failed"]]
    raw_space_count = len(preprocessing_artifacts.get("per_raw_space_objs", []))
    raw_opening_count = len(preprocessing_artifacts.get("per_raw_opening_objs", []))
    normalized_space_count = len(preprocessing_artifacts.get("per_space_objs", []))
    normalized_opening_count = len(preprocessing_artifacts.get("per_opening_objs", []))
    surface_entities = _build_surface_entities(
        external_shell.get("surfaces", []),
        min_area_threshold_m2=min_area_threshold_m2,
    )
    opening_surface_entities = _build_opening_surface_entities(opening_integration.get("opening_surfaces", []))

    ib_payload = internal_boundaries_payload or {}
    internal_boundary_entities = _build_internal_boundary_entities(
        ib_payload.get("oriented_surfaces", []),
        proximity_threshold_m=proximity_threshold_m,
    )

    clash_groups = list(preflight.get("clash_groups", []))

    return {
        "job_id": job_id,
        "schema": output_payload.get("schema"),
        "unit": preprocessing.get("unit", "meter"),
        "min_area_threshold_m2": min_area_threshold_m2,
        "proximity_threshold_m": proximity_threshold_m,
        "summary": {
            "space_count": output_payload.get("summary", {}).get("number_of_spaces", 0),
            "opening_count": output_payload.get("summary", {}).get("number_of_openings", 0),
            "failed_count": len(failed_entities),
            "valid_count": sum(1 for entity in entities if not entity["failed"]),
            "surface_count": len(surface_entities),
            "opening_surface_count": len(opening_surface_entities),
            "internal_boundary_count": len(internal_boundary_entities),
            "unclassified_surface_count": sum(
                1 for surface in surface_entities if surface.get("classification") == "unclassified"
            ),
            "clash_group_count": len(clash_groups),
        },
        "layers": {
            "raw_ifc_preview": {
                "available": raw_space_count > 0 or raw_opening_count > 0,
                "spaces_obj": preprocessing_artifacts.get("raw_spaces_all") if raw_space_count > 0 else None,
                "openings_obj": preprocessing_artifacts.get("raw_openings") if raw_opening_count > 0 else None,
                "glb": None,
            },
            "normalized_spaces": {
                "available": normalized_space_count > 0 and bool(preprocessing_artifacts.get("spaces_all")),
                "obj": preprocessing_artifacts.get("spaces_all") if normalized_space_count > 0 else None,
                "glb": None,
            },
            "openings": {
                "available": normalized_opening_count > 0 and bool(preprocessing_artifacts.get("openings")),
                "obj": preprocessing_artifacts.get("openings") if normalized_opening_count > 0 else None,
                "glb": None,
            },
            "failed_entities": {
                "available": bool(failed_entities),
                "count": len(failed_entities),
            },
            "envelope_shell": {
                "available": bool(external_shell_artifacts.get("shell_obj")),
                "obj": external_shell_artifacts.get("shell_obj"),
                "glb": None,
            },
            "surface_classification": {
                "available": len(surface_entities) > 0 and bool(external_shell_artifacts.get("surfaces_all")),
                "obj": external_shell_artifacts.get("surfaces_all") if len(surface_entities) > 0 else None,
                "glb": None,
                "count": len(surface_entities),
            },
            "opening_integration": {
                "available": len(opening_surface_entities) > 0 and bool(opening_integration_artifacts.get("obj")),
                "obj": opening_integration_artifacts.get("obj") if len(opening_surface_entities) > 0 else None,
                "glb": None,
                "count": len(opening_surface_entities),
            },
            "internal_boundaries": {
                "available": len(internal_boundary_entities) > 0 and bool(internal_boundaries_artifacts.get("obj")),
                "obj": internal_boundaries_artifacts.get("obj") if len(internal_boundary_entities) > 0 else None,
                "glb": None,
                "count": len(internal_boundary_entities),
            },
        },
        "clash_review": {
            "review_required": preflight.get("review_required", False),
            "resolution_status": preflight.get("resolution_status"),
            "recommended_resolution": preflight.get("recommended_resolution"),
            "clash_groups": clash_groups,
        },
        "entities": entities,
        "surface_entities": surface_entities,
        "opening_surface_entities": opening_surface_entities,
        "internal_boundary_entities": internal_boundary_entities,
    }


def _build_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_entities: list[dict[str, Any]] = []

    for entity in entities:
        global_id = entity.get("global_id")
        express_id = entity["express_id"]
        artifacts = entity.get("artifacts", {})
        object_name = global_id or f"entity_{express_id}"
        failed = not entity.get("valid_solid", False) or entity.get("preflight_failed", False)

        manifest_entities.append(
            {
                "object_name": object_name,
                "selection_type": "entity",
                "global_id": global_id,
                "express_id": express_id,
                "name": entity.get("name"),
                "entity_type": entity["entity_type"],
                "storey": entity.get("storey"),
                "building": entity.get("building"),
                "placement": entity.get("placement"),
                "valid": entity.get("valid_solid", False) and not entity.get("preflight_failed", False),
                "failed": failed,
                "reason": entity.get("preflight_reason") or entity.get("geometry_error"),
                "has_representation": entity.get("has_representation", False),
                "volume_m3": entity.get("volume_m3", 0.0),
                "face_count": entity.get("face_count", 0),
                "vertex_count": entity.get("vertex_count", 0),
                "component_count": entity.get("component_count", 0),
                "repair_backend": entity.get("repair_backend"),
                "repair_status": entity.get("repair_status"),
                "repair_reason": entity.get("repair_reason"),
                "repair_actions": list(entity.get("repair_actions", [])),
                "clash_groups": list(entity.get("clash_groups", [])),
                "clash_group_ids": list(entity.get("clash_group_ids", [])),
                "recommended_clash_action": entity.get("recommended_clash_action"),
                "marker_origin": _marker_origin(entity.get("placement")),
                "artifacts": {
                    "raw_obj": artifacts.get("raw_obj"),
                    "normalized_obj": artifacts.get("normalized_obj"),
                    "glb": artifacts.get("glb"),
                },
            }
        )

    return manifest_entities


def _build_surface_entities(
    surfaces: list[dict[str, Any]],
    *,
    min_area_threshold_m2: float = DEFAULT_MIN_AREA_THRESHOLD_M2,
) -> list[dict[str, Any]]:
    manifest_surfaces: list[dict[str, Any]] = []
    for surface in surfaces:
        surface_id = surface.get("surface_id") or surface.get("object_name")
        classification = surface.get("classification") or "unclassified"
        surface_artifacts = surface.get("artifacts", {})
        area = surface.get("area_m2", 0.0)
        manifest_surfaces.append(
            {
                "object_name": surface_id,
                "selection_type": "surface",
                "surface_id": surface_id,
                "name": surface.get("space_name") or surface_id,
                "entity_type": "SpaceSurface",
                "space_global_id": surface.get("space_global_id"),
                "space_express_id": surface.get("space_express_id"),
                "classification": classification,
                "failed": classification == "unclassified",
                "valid": classification != "unclassified",
                "reason": surface.get("reason"),
                "area_m2": area,
                "below_area_threshold": area < min_area_threshold_m2,
                "normal": surface.get("normal"),
                "centroid": surface.get("centroid"),
                "marker_origin": surface.get("centroid"),
                "artifacts": {
                    "classified_obj": surface_artifacts.get("classified_obj"),
                    "class_obj": surface_artifacts.get("class_obj"),
                    "shell_obj": surface_artifacts.get("shell_obj"),
                },
            }
        )
    return manifest_surfaces


def _marker_origin(placement: dict[str, Any] | None) -> list[float] | None:
    if not placement or not placement.get("available"):
        return None
    origin = placement.get("origin")
    if isinstance(origin, list) and len(origin) == 3:
        return origin
    return None


def _build_opening_surface_entities(surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_surfaces: list[dict[str, Any]] = []
    for surface in surfaces:
        surface_id = surface.get("surface_id", "")
        manifest_surfaces.append(
            {
                "object_name": surface_id,
                "selection_type": "opening_surface",
                "surface_id": surface_id,
                "name": surface.get("opening_name") or surface_id,
                "entity_type": "OpeningSurface",
                "boundary_type": surface.get("boundary_type"),
                "boundary_surface_id": surface.get("boundary_surface_id"),
                "boundary_classification": surface.get("boundary_classification"),
                "opening_express_id": surface.get("opening_express_id"),
                "opening_global_id": surface.get("opening_global_id"),
                "space_global_id": surface.get("space_global_id"),
                "space_express_id": surface.get("space_express_id"),
                "classification": "opening",
                "failed": False,
                "valid": True,
                "area_m2": surface.get("area_m2", 0.0),
                "normal": surface.get("normal"),
                "centroid": surface.get("centroid"),
                "marker_origin": surface.get("centroid"),
            }
        )
    return manifest_surfaces


def _build_internal_boundary_entities(
    oriented_surfaces: list[dict[str, Any]],
    *,
    proximity_threshold_m: float = DEFAULT_PROXIMITY_THRESHOLD_M,
) -> list[dict[str, Any]]:
    surfaces_by_id: dict[str, dict[str, Any]] = {
        s["oriented_surface_id"]: s for s in oriented_surfaces if s.get("oriented_surface_id")
    }

    manifest_surfaces: list[dict[str, Any]] = []
    for surface in oriented_surfaces:
        surface_id = surface.get("oriented_surface_id") or surface.get("object_name", "")
        paired_id = surface.get("paired_surface_id")

        thickness_m = _compute_pair_thickness(surface, surfaces_by_id.get(paired_id)) if paired_id else None

        manifest_surfaces.append(
            {
                "object_name": surface_id,
                "selection_type": "internal_boundary",
                "surface_id": surface_id,
                "name": surface.get("space_name") or surface_id,
                "entity_type": "InternalBoundary",
                "space_global_id": surface.get("space_global_id"),
                "space_express_id": surface.get("space_express_id"),
                "adjacent_space_global_id": surface.get("adjacent_space_global_id"),
                "adjacent_space_express_id": surface.get("adjacent_space_express_id"),
                "paired_surface_id": paired_id,
                "shared_surface_id": surface.get("shared_surface_id"),
                "classification": "internal",
                "failed": False,
                "valid": True,
                "area_m2": surface.get("area_m2", 0.0),
                "normal": surface.get("plane_normal"),
                "centroid": surface.get("centroid"),
                "marker_origin": surface.get("centroid"),
                "thickness_m": thickness_m,
                "proximity_conflict": thickness_m is not None and thickness_m < proximity_threshold_m,
            }
        )
    return manifest_surfaces


def _compute_pair_thickness(
    surface_a: dict[str, Any],
    surface_b: dict[str, Any] | None,
) -> float | None:
    if surface_b is None:
        return None
    point_a = surface_a.get("plane_point")
    point_b = surface_b.get("plane_point")
    normal_a = surface_a.get("plane_normal")
    if not point_a or not point_b or not normal_a:
        return None
    if len(point_a) != 3 or len(point_b) != 3 or len(normal_a) != 3:
        return None
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dz = point_b[2] - point_a[2]
    return abs(dx * normal_a[0] + dy * normal_a[1] + dz * normal_a[2])
