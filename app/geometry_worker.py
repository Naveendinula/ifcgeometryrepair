from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import ifcopenshell.geom
import ifcopenshell.util.shape as shape_util

from .ifc_extractor import ExtractedEntity, PreparedIFC
from .mesh_normalizer import build_obj_text, normalize_mesh


ShapeBuilder = Callable[[Any, Any], Any]
ExactRepairMode = Literal["disabled", "preferred"]
EXACT_REPAIR_CONTRACT_VERSION = 2


@dataclass(slots=True)
class GeometryPreprocessingResult:
    request: dict[str, Any]
    result: dict[str, Any]
    summary: dict[str, Any]


def build_shape(settings: Any, entity: Any) -> Any:
    return ifcopenshell.geom.create_shape(settings, entity)


def run_geometry_preprocessing(
    job_id: str,
    job_dir: Path,
    prepared_ifc: PreparedIFC,
    *,
    exact_repair_mode: ExactRepairMode = "preferred",
    exact_repair_worker_binary: Path | None = None,
    shape_builder: ShapeBuilder | None = None,
) -> GeometryPreprocessingResult:
    shape_builder = shape_builder or build_shape
    geometry_dir = job_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    request_payload = _build_request_payload(job_id, prepared_ifc, shape_builder)
    artifacts = _build_artifact_paths()
    request_path = job_dir / artifacts["request"]
    result_path = job_dir / artifacts["result"]
    summary_path = job_dir / artifacts["summary"]
    repair_request_path = job_dir / artifacts["repair_request"]
    repair_response_path = job_dir / artifacts["repair_response"]
    repair_report_path = job_dir / artifacts["repair_report"]
    _write_json(request_path, request_payload)

    repair_request_payload = _build_repair_request_payload(job_id, request_payload)
    _write_json(repair_request_path, repair_request_payload)

    result_payload, repair_payload, repair_response_payload = _run_preprocessing_pipeline(
        request_payload,
        repair_request_payload,
        exact_repair_mode=exact_repair_mode,
        exact_repair_worker_binary=exact_repair_worker_binary,
        repair_request_path=repair_request_path,
        repair_response_path=repair_response_path,
    )
    _write_json(repair_response_path, repair_response_payload)
    _write_json(repair_report_path, repair_payload)

    _write_debug_geometry(job_dir, request_payload["entities"], result_payload["entities"], artifacts)

    worker_backend = _resolve_worker_backend(result_payload["entities"])
    result_payload["worker_backend"] = worker_backend
    result_payload["unit"] = "meter"
    result_payload["source_unit_scale_to_meters"] = prepared_ifc.unit_scale_to_meters
    result_payload["repair"] = repair_payload
    result_payload["artifacts"] = artifacts
    result_payload["summary"] = _build_result_summary(result_payload["entities"], artifacts, repair_payload)
    _write_json(result_path, result_payload)

    summary_payload = _build_geometry_summary(job_id, result_payload)
    _write_json(summary_path, summary_payload)

    return GeometryPreprocessingResult(
        request=request_payload,
        result=result_payload,
        summary=summary_payload,
    )


def _build_request_payload(
    job_id: str,
    prepared_ifc: PreparedIFC,
    shape_builder: ShapeBuilder,
) -> dict[str, Any]:
    entities_payload: list[dict[str, Any]] = []

    for extracted in prepared_ifc.iter_entities():
        entities_payload.append(
            _serialize_request_entity(
                extracted,
                prepared_ifc.geometry_settings,
                prepared_ifc.unit_scale_to_meters,
                shape_builder,
            )
        )

    return {
        "job_id": job_id,
        "schema": prepared_ifc.schema,
        "unit": "meter",
        "source_unit_scale_to_meters": prepared_ifc.unit_scale_to_meters,
        "entities": entities_payload,
    }


def _serialize_request_entity(
    extracted: ExtractedEntity,
    geometry_settings: Any,
    unit_scale_to_meters: float,
    shape_builder: ShapeBuilder,
) -> dict[str, Any]:
    object_name = _entity_object_name(extracted.record.get("global_id"), extracted.record["express_id"])
    entity_payload = {
        "object_name": object_name,
        "global_id": extracted.record.get("global_id"),
        "express_id": extracted.record["express_id"],
        "name": extracted.record.get("name"),
        "entity_type": extracted.record["entity_type"],
        "has_representation": extracted.record["has_representation"],
        "storey": extracted.record.get("storey"),
        "building": extracted.record.get("building"),
        "placement": extracted.record.get("placement"),
        "source_unit_scale_to_meters": unit_scale_to_meters,
        "mesh": None,
        "skip_reason": None,
    }

    if not extracted.record["has_representation"]:
        entity_payload["skip_reason"] = "Missing representation"
        return entity_payload

    try:
        shape = shape_builder(geometry_settings, extracted.product)
        vertices = shape_util.get_vertices(shape.geometry).tolist()
        faces = shape_util.get_faces(shape.geometry).tolist()
    except Exception as exc:
        entity_payload["skip_reason"] = f"Tessellation failed: {exc}"
        return entity_payload

    if not vertices or not faces:
        entity_payload["skip_reason"] = "Tessellation produced an empty mesh"
        return entity_payload

    entity_payload["mesh"] = {
        "vertices": vertices,
        "faces": faces,
    }
    return entity_payload


def _run_preprocessing_pipeline(
    request_payload: dict[str, Any],
    repair_request_payload: dict[str, Any],
    *,
    exact_repair_mode: ExactRepairMode,
    exact_repair_worker_binary: Path | None,
    repair_request_path: Path,
    repair_response_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    exact_repair_results_by_express_id: dict[int, dict[str, Any]] = {}
    exact_repair_worker_invoked = False
    exact_repair_fallback_reason = None
    exact_repair_response_payload = _build_default_repair_response_payload(
        status="not_invoked",
        reason="Exact repair was not requested for this run.",
    )

    repair_candidates = [
        entity
        for entity in request_payload["entities"]
        if entity["entity_type"] == "IfcSpace" and not entity.get("skip_reason") and entity.get("mesh")
    ]
    resolved_exact_repair_worker_binary = (
        exact_repair_worker_binary if exact_repair_worker_binary and exact_repair_worker_binary.exists() else None
    )

    if exact_repair_mode == "preferred" and repair_candidates and resolved_exact_repair_worker_binary is not None:
        exact_repair_worker_invoked = True
        try:
            _invoke_exact_repair_worker(resolved_exact_repair_worker_binary, repair_request_path, repair_response_path)
            candidate_response_payload = _read_json(repair_response_path)
            exact_repair_results_by_express_id = _parse_exact_repair_response(candidate_response_payload)
            exact_repair_response_payload = candidate_response_payload
        except Exception as exc:
            exact_repair_fallback_reason = f"Exact repair worker fallback: {exc}"
            exact_repair_response_payload = _build_default_repair_response_payload(
                status="failed",
                reason=exact_repair_fallback_reason,
                worker_backend="cpp-cgal",
            )
    elif exact_repair_mode == "disabled":
        exact_repair_fallback_reason = "Exact repair disabled by server settings."
        exact_repair_response_payload = _build_default_repair_response_payload(
            status="disabled",
            reason=exact_repair_fallback_reason,
        )
    elif repair_candidates and resolved_exact_repair_worker_binary is None:
        exact_repair_fallback_reason = "Exact repair worker unavailable."
        exact_repair_response_payload = _build_default_repair_response_payload(
            status="unavailable",
            reason=exact_repair_fallback_reason,
        )
    elif not repair_candidates:
        exact_repair_response_payload = _build_default_repair_response_payload(
            status="idle",
            reason="No valid IfcSpace meshes were available for exact repair.",
        )

    for entity in request_payload["entities"]:
        if entity.get("skip_reason"):
            entities.append(
                _build_skipped_entity_result(
                    entity,
                    repair_backend="none",
                    repair_status="not_attempted",
                    repair_reason=entity["skip_reason"],
                )
            )
            continue

        if entity["entity_type"] == "IfcOpeningElement":
            entities.append(
                _normalize_entity_with_python(
                    entity,
                    repair_backend="none",
                    repair_status="not_attempted",
                    repair_reason="Exact repair targets IfcSpace only.",
                )
            )
            continue

        exact_result = exact_repair_results_by_express_id.get(entity["express_id"])
        if exact_result is not None and exact_result.get("valid") and _mesh_available(exact_result):
            entities.append(_build_exact_entity_result(entity, exact_result))
            continue

        fallback_reason = (
            exact_result.get("repair_reason")
            if isinstance(exact_result, dict)
            else None
        ) or (
            exact_result.get("reason")
            if isinstance(exact_result, dict)
            else None
        ) or exact_repair_fallback_reason or "Exact repair result unavailable for space."
        entities.append(
            _normalize_entity_with_python(
                entity,
                repair_backend="python",
                repair_status="fallback_python",
                repair_reason=fallback_reason,
                extra_actions=[f"exact_repair_fallback:{_slugify(fallback_reason)}"],
            )
        )

    repair_payload = _build_repair_report(
        request_payload,
        entities,
        exact_repair_mode=exact_repair_mode,
        exact_repair_worker_binary=exact_repair_worker_binary,
        exact_repair_worker_invoked=exact_repair_worker_invoked,
        exact_repair_response_payload=exact_repair_response_payload,
        exact_repair_fallback_reason=exact_repair_fallback_reason,
    )
    return {
        "job_id": request_payload["job_id"],
        "schema": request_payload["schema"],
        "entities": entities,
    }, repair_payload, exact_repair_response_payload


def _build_repair_request_payload(job_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": EXACT_REPAIR_CONTRACT_VERSION,
        "job_id": job_id,
        "unit": request_payload["unit"],
        "source_unit_scale_to_meters": request_payload["source_unit_scale_to_meters"],
        "spaces": [
            {
                "object_name": entity["object_name"],
                "global_id": entity.get("global_id"),
                "express_id": entity["express_id"],
                "name": entity.get("name"),
                "mesh": entity.get("mesh"),
            }
            for entity in request_payload["entities"]
            if entity["entity_type"] == "IfcSpace" and not entity.get("skip_reason") and entity.get("mesh")
        ],
    }


def _build_default_repair_response_payload(
    *,
    status: str,
    reason: str,
    worker_backend: str | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": EXACT_REPAIR_CONTRACT_VERSION,
        "status": status,
        "worker_backend": worker_backend,
        "reason": reason,
        "spaces": [],
    }


def _parse_exact_repair_response(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if payload.get("contract_version") != EXACT_REPAIR_CONTRACT_VERSION:
        raise RuntimeError("Exact repair worker returned an unsupported contract version.")
    spaces = payload.get("spaces")
    if not isinstance(spaces, list):
        raise RuntimeError("Exact repair worker response is missing the spaces list.")

    results_by_express_id: dict[int, dict[str, Any]] = {}
    for entry in spaces:
        if not isinstance(entry, dict):
            continue
        express_id = entry.get("express_id")
        if not isinstance(express_id, int):
            continue
        results_by_express_id[express_id] = entry
    return results_by_express_id


def _entity_result_base(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_name": entity["object_name"],
        "global_id": entity.get("global_id"),
        "express_id": entity["express_id"],
        "name": entity.get("name"),
        "entity_type": entity["entity_type"],
        "has_representation": entity["has_representation"],
    }


def _build_skipped_entity_result(
    entity: dict[str, Any],
    *,
    repair_backend: str,
    repair_status: str,
    repair_reason: str | None,
) -> dict[str, Any]:
    result_entity = _entity_result_base(entity)
    result_entity.update(
        {
            "mesh": None,
            "vertex_count": 0,
            "face_count": 0,
            "component_count": 0,
            "components": [],
            "repair_actions": [],
            "repair_backend": repair_backend,
            "repair_status": repair_status,
            "repair_reason": repair_reason,
            "closed": False,
            "manifold": False,
            "outward_normals": False,
            "volume_m3": 0.0,
            "valid": False,
            "reason": entity.get("skip_reason"),
            "artifacts": {},
        }
    )
    return result_entity


def _normalize_entity_with_python(
    entity: dict[str, Any],
    *,
    repair_backend: str,
    repair_status: str,
    repair_reason: str | None,
    extra_actions: list[str] | None = None,
) -> dict[str, Any]:
    result_entity = _entity_result_base(entity)
    normalized = normalize_mesh(
        entity["mesh"]["vertices"],
        entity["mesh"]["faces"],
    )
    normalized["repair_actions"] = [*(extra_actions or []), *normalized.get("repair_actions", [])]
    normalized["repair_backend"] = repair_backend
    normalized["repair_status"] = repair_status
    normalized["repair_reason"] = repair_reason
    normalized["artifacts"] = {}
    result_entity.update(normalized)
    return result_entity


def _build_exact_entity_result(entity: dict[str, Any], exact_result: dict[str, Any]) -> dict[str, Any]:
    result_entity = _entity_result_base(entity)
    result_entity.update(
        {
            "mesh": exact_result.get("mesh"),
            "vertex_count": int(exact_result.get("vertex_count", 0)),
            "face_count": int(exact_result.get("face_count", 0)),
            "component_count": int(exact_result.get("component_count", 0)),
            "components": list(exact_result.get("components", [])),
            "repair_actions": list(exact_result.get("repair_actions", [])),
            "repair_backend": exact_result.get("repair_backend", "cpp-cgal"),
            "repair_status": exact_result.get("repair_status", "exact_repaired"),
            "repair_reason": exact_result.get("repair_reason"),
            "closed": bool(exact_result.get("closed")),
            "manifold": bool(exact_result.get("manifold")),
            "outward_normals": bool(exact_result.get("outward_normals")),
            "volume_m3": float(exact_result.get("volume_m3", 0.0)),
            "valid": bool(exact_result.get("valid")),
            "reason": exact_result.get("reason"),
            "artifacts": {},
        }
    )
    return result_entity


def _resolve_worker_backend(entities: list[dict[str, Any]]) -> str:
    if any(entity.get("repair_backend") == "cpp-cgal" for entity in entities):
        return "hybrid"
    return "python"


def _build_repair_report(
    request_payload: dict[str, Any],
    result_entities: list[dict[str, Any]],
    *,
    exact_repair_mode: ExactRepairMode,
    exact_repair_worker_binary: Path | None,
    exact_repair_worker_invoked: bool,
    exact_repair_response_payload: dict[str, Any],
    exact_repair_fallback_reason: str | None,
) -> dict[str, Any]:
    repair_entries = [
        {
            "global_id": entity.get("global_id"),
            "express_id": entity["express_id"],
            "name": entity.get("name"),
            "repair_backend": entity.get("repair_backend"),
            "repair_status": entity.get("repair_status"),
            "repair_reason": entity.get("repair_reason"),
            "repair_actions": list(entity.get("repair_actions", [])),
            "valid": bool(entity.get("valid")),
        }
        for entity in result_entities
        if entity["entity_type"] == "IfcSpace"
    ]
    worker_available = bool(exact_repair_worker_binary and exact_repair_worker_binary.exists())
    if exact_repair_worker_invoked:
        effective_mode = "exact_repair"
    elif exact_repair_mode == "disabled":
        effective_mode = "disabled"
    elif repair_entries:
        effective_mode = "python_fallback"
    else:
        effective_mode = "idle"

    return {
        "contract_version": EXACT_REPAIR_CONTRACT_VERSION,
        "mode_requested": exact_repair_mode,
        "effective_mode": effective_mode,
        "worker_path": str(exact_repair_worker_binary) if exact_repair_worker_binary else None,
        "worker_available": worker_available,
        "worker_invoked": exact_repair_worker_invoked,
        "worker_backend": exact_repair_response_payload.get("worker_backend"),
        "response_status": exact_repair_response_payload.get("status"),
        "fallback_reason": exact_repair_fallback_reason,
        "summary": {
            "effective_mode": effective_mode,
            "space_count": len(repair_entries),
            "attempted_space_count": len(exact_repair_response_payload.get("spaces", []))
            if exact_repair_worker_invoked
            else 0,
            "exact_repaired_space_count": sum(1 for entry in repair_entries if entry["repair_status"] == "exact_repaired"),
            "exact_passthrough_space_count": sum(
                1 for entry in repair_entries if entry["repair_status"] == "exact_passthrough"
            ),
            "python_fallback_space_count": sum(1 for entry in repair_entries if entry["repair_status"] == "fallback_python"),
            "not_attempted_space_count": sum(1 for entry in repair_entries if entry["repair_status"] == "not_attempted"),
        },
        "spaces": repair_entries,
    }


def _mesh_available(entity: dict[str, Any]) -> bool:
    mesh = entity.get("mesh") or {}
    return bool(mesh.get("vertices") and mesh.get("faces"))


def _invoke_exact_repair_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
    completed = subprocess.run(
        [str(worker_binary), str(request_path), str(result_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "exact repair worker failed"
        raise RuntimeError(stderr)


def _write_debug_geometry(
    job_dir: Path,
    request_entities: list[dict[str, Any]],
    result_entities: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> None:
    raw_entity_artifacts = _write_entity_geometry(
        job_dir,
        request_entities,
        directory_prefix="geometry/raw",
        spaces_all_relative=artifacts["raw_spaces_all"],
        openings_all_relative=artifacts["raw_openings"],
        per_space_list=artifacts["per_raw_space_objs"],
        per_opening_list=artifacts["per_raw_opening_objs"],
    )
    normalized_entity_artifacts = _write_entity_geometry(
        job_dir,
        result_entities,
        directory_prefix="geometry",
        spaces_all_relative=artifacts["spaces_all"],
        openings_all_relative=artifacts["openings"],
        per_space_list=artifacts["per_space_objs"],
        per_opening_list=artifacts["per_opening_objs"],
    )

    for entity in result_entities:
        express_id = entity["express_id"]
        raw_obj = raw_entity_artifacts.get(express_id)
        normalized_obj = normalized_entity_artifacts.get(express_id)
        entity["artifacts"] = {
            "obj": normalized_obj or raw_obj,
            "normalized_obj": normalized_obj,
            "raw_obj": raw_obj,
            "glb": None,
        }

    preview_path = job_dir / artifacts["preview"]
    if artifacts["per_space_objs"]:
        shutil.copyfile(job_dir / artifacts["spaces_all"], preview_path)
    elif artifacts["per_opening_objs"]:
        shutil.copyfile(job_dir / artifacts["openings"], preview_path)
    elif artifacts["per_raw_space_objs"]:
        shutil.copyfile(job_dir / artifacts["raw_spaces_all"], preview_path)
    elif artifacts["per_raw_opening_objs"]:
        shutil.copyfile(job_dir / artifacts["raw_openings"], preview_path)
    else:
        _write_text(preview_path, "# No geometry available\n")


def _write_entity_geometry(
    job_dir: Path,
    entities: list[dict[str, Any]],
    *,
    directory_prefix: str,
    spaces_all_relative: str,
    openings_all_relative: str,
    per_space_list: list[str],
    per_opening_list: list[str],
) -> dict[int, str]:
    spaces_dir = job_dir / directory_prefix / "spaces"
    openings_dir = job_dir / directory_prefix / "openings"
    spaces_dir.mkdir(parents=True, exist_ok=True)
    openings_dir.mkdir(parents=True, exist_ok=True)

    space_meshes: list[dict[str, Any]] = []
    opening_meshes: list[dict[str, Any]] = []
    entity_artifacts: dict[int, str] = {}

    for entity in entities:
        mesh = entity.get("mesh")
        if mesh is None or not mesh.get("faces"):
            continue

        mesh_payload = {
            "name": entity["object_name"],
            "vertices": mesh["vertices"],
            "faces": mesh["faces"],
        }

        if entity["entity_type"] == "IfcSpace":
            relative_path = f"{directory_prefix}/spaces/{_entity_filename(entity['object_name'])}.obj"
            _write_text(job_dir / relative_path, build_obj_text([mesh_payload]))
            space_meshes.append(mesh_payload)
            per_space_list.append(relative_path)
            entity_artifacts[entity["express_id"]] = relative_path
        elif entity["entity_type"] == "IfcOpeningElement":
            relative_path = f"{directory_prefix}/openings/{_entity_filename(entity['object_name'])}.obj"
            _write_text(job_dir / relative_path, build_obj_text([mesh_payload]))
            opening_meshes.append(mesh_payload)
            per_opening_list.append(relative_path)
            entity_artifacts[entity["express_id"]] = relative_path

    _write_aggregate_obj(job_dir / spaces_all_relative, space_meshes)
    _write_aggregate_obj(job_dir / openings_all_relative, opening_meshes)
    return entity_artifacts


def _write_aggregate_obj(path: Path, meshes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(path, build_obj_text(meshes))


def _build_artifact_paths() -> dict[str, Any]:
    return {
        "request": "geometry/request.json",
        "result": "geometry/result.json",
        "summary": "geometry/geometry_summary.json",
        "repair_request": "geometry/repair_request.json",
        "repair_response": "geometry/repair_response.json",
        "repair_report": "geometry/repair_report.json",
        "viewer_manifest": "geometry/viewer_manifest.json",
        "raw_spaces_all": "geometry/raw/spaces_all.obj",
        "raw_openings": "geometry/raw/openings.obj",
        "spaces_all": "geometry/spaces_all.obj",
        "openings": "geometry/openings.obj",
        "per_raw_space_objs": [],
        "per_raw_opening_objs": [],
        "per_space_objs": [],
        "per_opening_objs": [],
        "preview": "preview.obj",
    }


def _build_result_summary(
    entities: list[dict[str, Any]],
    artifacts: dict[str, Any],
    repair_payload: dict[str, Any],
) -> dict[str, Any]:
    spaces = [entity for entity in entities if entity["entity_type"] == "IfcSpace"]
    openings = [entity for entity in entities if entity["entity_type"] == "IfcOpeningElement"]
    entities_with_mesh = [entity for entity in entities if entity.get("mesh") is not None]
    valid_entities = [entity for entity in entities if entity["valid"]]
    invalid_entities = [entity for entity in entities if not entity["valid"]]

    return {
        "entities_total": len(entities),
        "entities_with_input_mesh": len(entities_with_mesh),
        "normalized_entities": len(entities_with_mesh),
        "valid_entities": len(valid_entities),
        "invalid_entities": len(invalid_entities),
        "space_count": len(spaces),
        "opening_count": len(openings),
        "valid_spaces": sum(1 for entity in spaces if entity["valid"]),
        "invalid_spaces": sum(1 for entity in spaces if not entity["valid"]),
        "valid_openings": sum(1 for entity in openings if entity["valid"]),
        "invalid_openings": sum(1 for entity in openings if not entity["valid"]),
        "spaces_with_obj_exports": len(artifacts["per_space_objs"]),
        "openings_with_obj_exports": len(artifacts["per_opening_objs"]),
        "repair_effective_mode": repair_payload["summary"]["effective_mode"],
        "repair_exact_repaired_spaces": repair_payload["summary"]["exact_repaired_space_count"],
        "repair_exact_passthrough_spaces": repair_payload["summary"]["exact_passthrough_space_count"],
        "repair_python_fallback_spaces": repair_payload["summary"]["python_fallback_space_count"],
        "repair_not_attempted_spaces": repair_payload["summary"]["not_attempted_space_count"],
    }


def _build_geometry_summary(job_id: str, result_payload: dict[str, Any]) -> dict[str, Any]:
    invalid_entities = [
        {
            "object_name": entity["object_name"],
            "global_id": entity.get("global_id"),
            "express_id": entity["express_id"],
            "name": entity.get("name"),
            "entity_type": entity["entity_type"],
            "reason": entity.get("reason"),
            "repair_backend": entity.get("repair_backend"),
            "repair_status": entity.get("repair_status"),
            "repair_reason": entity.get("repair_reason"),
            "valid": entity["valid"],
        }
        for entity in result_payload["entities"]
        if not entity["valid"]
    ]

    return {
        "job_id": job_id,
        "worker_backend": result_payload["worker_backend"],
        "unit": result_payload["unit"],
        "source_unit_scale_to_meters": result_payload["source_unit_scale_to_meters"],
        "summary": result_payload["summary"],
        "repair": result_payload["repair"],
        "invalid_entities": invalid_entities,
        "artifacts": result_payload["artifacts"],
        "entities": [
            {
                "object_name": entity["object_name"],
                "global_id": entity.get("global_id"),
                "express_id": entity["express_id"],
                "name": entity.get("name"),
                "entity_type": entity["entity_type"],
                "valid": entity["valid"],
                "closed": entity["closed"],
                "manifold": entity["manifold"],
                "outward_normals": entity["outward_normals"],
                "volume_m3": entity["volume_m3"],
                "face_count": entity["face_count"],
                "vertex_count": entity["vertex_count"],
                "component_count": entity["component_count"],
                "components": entity["components"],
                "repair_backend": entity.get("repair_backend"),
                "repair_status": entity.get("repair_status"),
                "repair_reason": entity.get("repair_reason"),
                "repair_actions": entity["repair_actions"],
                "reason": entity.get("reason"),
                "artifacts": entity.get("artifacts", {}),
            }
            for entity in result_payload["entities"]
        ],
    }


def _entity_object_name(global_id: str | None, express_id: int) -> str:
    return global_id or f"entity_{express_id}"


def _entity_filename(object_name: str) -> str:
    return object_name.replace("/", "_").replace("\\", "_")


def _slugify(value: str) -> str:
    sanitized = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return sanitized.strip("_") or "fallback"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
