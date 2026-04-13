from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import ifcopenshell.util.placement as placement_util
import ifcopenshell.util.representation as representation_util
import ifcopenshell.util.unit as unit_util


@dataclass(slots=True)
class ParsedIFC:
    model: Any
    schema: str
    input_path: Path


@dataclass(slots=True)
class ExtractedEntity:
    product: Any
    record: dict[str, Any]


@dataclass(slots=True)
class PreparedIFC:
    model: Any
    schema: str
    spaces: list[ExtractedEntity]
    openings: list[ExtractedEntity]
    geometry_settings: Any
    unit_scale_to_meters: float

    def iter_entities(self) -> list[ExtractedEntity]:
        return [*self.spaces, *self.openings]


def parse_ifc_file(input_path: Path) -> ParsedIFC:
    model = ifcopenshell.open(str(input_path))
    return ParsedIFC(model=model, schema=str(model.schema), input_path=input_path)


def prepare_extraction(parsed_ifc: ParsedIFC) -> PreparedIFC:
    return PreparedIFC(
        model=parsed_ifc.model,
        schema=parsed_ifc.schema,
        spaces=[_extract_entity(entity) for entity in parsed_ifc.model.by_type("IfcSpace")],
        openings=[_extract_entity(entity) for entity in parsed_ifc.model.by_type("IfcOpeningElement")],
        geometry_settings=_build_geometry_settings(),
        unit_scale_to_meters=float(unit_util.calculate_unit_scale(parsed_ifc.model)),
    )


def build_extraction_report(
    job_id: str,
    prepared_ifc: PreparedIFC,
    preprocessing_result: dict[str, Any],
    internal_boundary_result: dict[str, Any],
) -> dict[str, Any]:
    geometry_by_express_id = {
        entity["express_id"]: entity for entity in preprocessing_result.get("entities", [])
    }

    spaces = [_merge_entity_record(extracted.record, geometry_by_express_id) for extracted in prepared_ifc.spaces]
    openings = [_merge_entity_record(extracted.record, geometry_by_express_id) for extracted in prepared_ifc.openings]

    missing_spaces = [
        _entity_ref_from_record(record) for record in spaces if not record["has_representation"]
    ]
    invalid_solids = [
        {
            **_entity_ref_from_record(record),
            "error": record["geometry_error"],
        }
        for record in [*spaces, *openings]
        if record["has_representation"] and not record["geometry_ok"]
    ]

    preprocessing_invalid_entities = [
        {
            "global_id": entity.get("global_id"),
            "express_id": entity["express_id"],
            "name": entity.get("name"),
            "entity_type": entity["entity_type"],
            "reason": entity.get("reason"),
            "valid": entity["valid"],
        }
        for entity in preprocessing_result.get("entities", [])
        if not entity["valid"]
    ]

    return {
        "success": True,
        "job_id": job_id,
        "schema": prepared_ifc.schema,
        "summary": {
            "number_of_spaces": len(spaces),
            "number_of_openings": len(openings),
            "entities_processed": len(spaces) + len(openings),
            "issues_found": len(missing_spaces) + len(invalid_solids),
        },
        "geometry_sanity": {
            "number_of_spaces": len(spaces),
            "number_of_openings": len(openings),
            "spaces_with_missing_representation": missing_spaces,
            "invalid_solids": invalid_solids,
        },
        "spaces": spaces,
        "openings": openings,
        "preprocessing": {
            "worker_backend": preprocessing_result.get("worker_backend"),
            "unit": preprocessing_result.get("unit"),
            "source_unit_scale_to_meters": preprocessing_result.get("source_unit_scale_to_meters"),
            "summary": preprocessing_result.get("summary", {}),
            "invalid_entities": preprocessing_invalid_entities,
            "artifacts": preprocessing_result.get("artifacts", {}),
        },
        "internal_boundaries": {
            "threshold_m": internal_boundary_result.get("threshold_m"),
            "summary": internal_boundary_result.get("summary", {}),
            "adjacencies": internal_boundary_result.get("adjacencies", []),
            "artifacts": internal_boundary_result.get("artifacts", {}),
        },
    }


def _build_geometry_settings() -> Any:
    settings = ifcopenshell.geom.settings()
    for option in ("use-world-coords", "weld-vertices", "reorient-shells", "unify-shapes", "validate"):
        settings.set(option, True)
    return settings


def _extract_entity(entity: Any) -> ExtractedEntity:
    storey, building = _resolve_spatial_refs(entity)
    record = {
        "express_id": entity.id(),
        "global_id": getattr(entity, "GlobalId", None),
        "name": getattr(entity, "Name", None),
        "entity_type": entity.is_a(),
        "storey": storey,
        "building": building,
        "placement": _serialize_placement(getattr(entity, "ObjectPlacement", None)),
        "has_representation": _has_representation(entity),
    }
    return ExtractedEntity(product=entity, record=record)


def _merge_entity_record(
    base_record: dict[str, Any],
    geometry_by_express_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    record = dict(base_record)
    geometry = geometry_by_express_id.get(record["express_id"])

    geometry_error = None
    geometry_ok = False
    if not record["has_representation"]:
        geometry_error = "Missing representation"
    elif geometry is None:
        geometry_error = "Geometry result unavailable"
    elif geometry["valid"]:
        geometry_ok = True
    else:
        geometry_error = geometry.get("reason")

    record.update(
        {
            "geometry_ok": geometry_ok,
            "geometry_error": geometry_error,
            "valid_solid": geometry["valid"] if geometry else False,
            "closed": geometry["closed"] if geometry else False,
            "manifold": geometry["manifold"] if geometry else False,
            "outward_normals": geometry["outward_normals"] if geometry else False,
            "volume_m3": geometry["volume_m3"] if geometry else 0.0,
            "face_count": geometry["face_count"] if geometry else 0,
            "vertex_count": geometry["vertex_count"] if geometry else 0,
            "component_count": geometry["component_count"] if geometry else 0,
            "repair_actions": geometry["repair_actions"] if geometry else [],
            "artifacts": geometry.get("artifacts", {}) if geometry else {},
        }
    )
    return record


def _resolve_spatial_refs(entity: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    storey = None
    building = None

    for ancestor in _iter_spatial_ancestors(entity):
        if storey is None and ancestor.is_a("IfcBuildingStorey"):
            storey = _serialize_context(ancestor)
        if building is None and ancestor.is_a("IfcBuilding"):
            building = _serialize_context(ancestor)
        if storey and building:
            break

    return storey, building


def _iter_spatial_ancestors(entity: Any):
    seen: set[int] = set()
    current = element_util.get_container(entity) or element_util.get_parent(entity)

    while current is not None and current.id() not in seen:
        seen.add(current.id())
        yield current
        current = element_util.get_parent(current)


def _serialize_context(entity: Any) -> dict[str, Any]:
    return {
        "express_id": entity.id(),
        "global_id": getattr(entity, "GlobalId", None),
        "name": getattr(entity, "Name", None),
        "entity_type": entity.is_a(),
    }


def _serialize_placement(object_placement: Any) -> dict[str, Any]:
    if object_placement is None:
        return {
            "available": False,
            "origin": None,
            "matrix": None,
        }

    try:
        matrix = placement_util.get_local_placement(object_placement).tolist()
        origin = [matrix[0][3], matrix[1][3], matrix[2][3]]
        return {
            "available": True,
            "origin": origin,
            "matrix": matrix,
        }
    except Exception:
        return {
            "available": False,
            "origin": None,
            "matrix": None,
        }


def _has_representation(entity: Any) -> bool:
    try:
        body_representation = representation_util.get_representation(
            entity,
            "Model",
            "Body",
            "MODEL_VIEW",
        )
        if body_representation is not None:
            return True
    except Exception:
        pass

    product_representation = getattr(entity, "Representation", None)
    representations = getattr(product_representation, "Representations", None) or []

    for representation in representations:
        try:
            if representation_util.resolve_representation(representation) is not None:
                return True
        except Exception:
            continue

    return False


def _entity_ref_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "express_id": record["express_id"],
        "global_id": record.get("global_id"),
        "name": record.get("name"),
        "entity_type": record["entity_type"],
    }
