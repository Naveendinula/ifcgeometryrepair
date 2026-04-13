from __future__ import annotations

from typing import Any


def build_viewer_manifest(job_id: str, output_payload: dict[str, Any]) -> dict[str, Any]:
    preprocessing = output_payload.get("preprocessing", {})
    artifacts = preprocessing.get("artifacts", {})
    entities = [*_build_entities(output_payload.get("spaces", [])), *_build_entities(output_payload.get("openings", []))]
    failed_entities = [entity for entity in entities if entity["failed"]]

    return {
        "job_id": job_id,
        "schema": output_payload.get("schema"),
        "unit": preprocessing.get("unit", "meter"),
        "summary": {
            "space_count": output_payload.get("summary", {}).get("number_of_spaces", 0),
            "opening_count": output_payload.get("summary", {}).get("number_of_openings", 0),
            "failed_count": len(failed_entities),
            "valid_count": sum(1 for entity in entities if not entity["failed"]),
        },
        "layers": {
            "raw_ifc_preview": {
                "available": bool(artifacts.get("raw_spaces_all") or artifacts.get("raw_openings")),
                "spaces_obj": artifacts.get("raw_spaces_all"),
                "openings_obj": artifacts.get("raw_openings"),
                "glb": None,
            },
            "normalized_spaces": {
                "available": bool(artifacts.get("spaces_all")),
                "obj": artifacts.get("spaces_all"),
                "glb": None,
            },
            "openings": {
                "available": bool(artifacts.get("openings")),
                "obj": artifacts.get("openings"),
                "glb": None,
            },
            "failed_entities": {
                "available": bool(failed_entities),
                "count": len(failed_entities),
            },
        },
        "entities": entities,
    }


def _build_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_entities: list[dict[str, Any]] = []

    for entity in entities:
        global_id = entity.get("global_id")
        express_id = entity["express_id"]
        artifacts = entity.get("artifacts", {})
        object_name = global_id or f"entity_{express_id}"
        failed = not entity.get("valid_solid", False)

        manifest_entities.append(
            {
                "object_name": object_name,
                "global_id": global_id,
                "express_id": express_id,
                "name": entity.get("name"),
                "entity_type": entity["entity_type"],
                "storey": entity.get("storey"),
                "building": entity.get("building"),
                "placement": entity.get("placement"),
                "valid": entity.get("valid_solid", False),
                "failed": failed,
                "reason": entity.get("geometry_error"),
                "has_representation": entity.get("has_representation", False),
                "volume_m3": entity.get("volume_m3", 0.0),
                "face_count": entity.get("face_count", 0),
                "vertex_count": entity.get("vertex_count", 0),
                "component_count": entity.get("component_count", 0),
                "marker_origin": _marker_origin(entity.get("placement")),
                "artifacts": {
                    "raw_obj": artifacts.get("raw_obj"),
                    "normalized_obj": artifacts.get("normalized_obj"),
                    "glb": artifacts.get("glb"),
                },
            }
        )

    return manifest_entities


def _marker_origin(placement: dict[str, Any] | None) -> list[float] | None:
    if not placement or not placement.get("available"):
        return None
    origin = placement.get("origin")
    if isinstance(origin, list) and len(origin) == 3:
        return origin
    return None
