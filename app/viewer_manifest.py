from __future__ import annotations

from typing import Any


def build_viewer_manifest(job_id: str, output_payload: dict[str, Any]) -> dict[str, Any]:
    preprocessing = output_payload.get("preprocessing", {})
    preprocessing_artifacts = preprocessing.get("artifacts", {})
    preflight = output_payload.get("preflight", {})
    external_shell = output_payload.get("external_shell", {})
    external_shell_artifacts = external_shell.get("artifacts", {})
    entities = [*_build_entities(output_payload.get("spaces", [])), *_build_entities(output_payload.get("openings", []))]
    failed_entities = [entity for entity in entities if entity["failed"]]
    surface_entities = _build_surface_entities(external_shell.get("surfaces", []))
    clash_groups = list(preflight.get("clash_groups", []))

    return {
        "job_id": job_id,
        "schema": output_payload.get("schema"),
        "unit": preprocessing.get("unit", "meter"),
        "summary": {
            "space_count": output_payload.get("summary", {}).get("number_of_spaces", 0),
            "opening_count": output_payload.get("summary", {}).get("number_of_openings", 0),
            "failed_count": len(failed_entities),
            "valid_count": sum(1 for entity in entities if not entity["failed"]),
            "surface_count": len(surface_entities),
            "unclassified_surface_count": sum(
                1 for surface in surface_entities if surface.get("classification") == "unclassified"
            ),
            "clash_group_count": len(clash_groups),
        },
        "layers": {
            "raw_ifc_preview": {
                "available": bool(preprocessing_artifacts.get("raw_spaces_all") or preprocessing_artifacts.get("raw_openings")),
                "spaces_obj": preprocessing_artifacts.get("raw_spaces_all"),
                "openings_obj": preprocessing_artifacts.get("raw_openings"),
                "glb": None,
            },
            "normalized_spaces": {
                "available": bool(preprocessing_artifacts.get("spaces_all")),
                "obj": preprocessing_artifacts.get("spaces_all"),
                "glb": None,
            },
            "openings": {
                "available": bool(preprocessing_artifacts.get("openings")),
                "obj": preprocessing_artifacts.get("openings"),
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
                "available": bool(external_shell_artifacts.get("surfaces_all")),
                "obj": external_shell_artifacts.get("surfaces_all"),
                "glb": None,
                "count": len(surface_entities),
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


def _build_surface_entities(surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_surfaces: list[dict[str, Any]] = []
    for surface in surfaces:
        surface_id = surface.get("surface_id") or surface.get("object_name")
        classification = surface.get("classification") or "unclassified"
        surface_artifacts = surface.get("artifacts", {})
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
                "area_m2": surface.get("area_m2", 0.0),
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
