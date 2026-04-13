from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import ifcopenshell.geom
import ifcopenshell.util.shape as shape_util

from .ifc_extractor import ExtractedEntity, PreparedIFC
from .mesh_normalizer import build_obj_text, normalize_mesh


ShapeBuilder = Callable[[Any, Any], Any]


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
    worker_binary: Path | None = None,
    shape_builder: ShapeBuilder | None = None,
) -> GeometryPreprocessingResult:
    shape_builder = shape_builder or build_shape
    geometry_dir = job_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    request_payload = _build_request_payload(job_id, prepared_ifc, shape_builder)
    request_path = geometry_dir / "request.json"
    result_path = geometry_dir / "result.json"
    summary_path = geometry_dir / "geometry_summary.json"
    _write_json(request_path, request_payload)

    resolved_worker_binary = worker_binary if worker_binary and worker_binary.exists() else None
    if resolved_worker_binary is None:
        result_payload = _run_python_worker(request_payload)
        worker_backend = "python"
    else:
        _invoke_cpp_worker(resolved_worker_binary, request_path, result_path)
        result_payload = _read_json(result_path)
        if not isinstance(result_payload.get("entities"), list):
            result_payload = _run_python_worker(request_payload)
            worker_backend = "python"
        else:
            worker_backend = "cpp"

    artifacts = _build_artifact_paths()
    _write_debug_geometry(job_dir, request_payload["entities"], result_payload["entities"], artifacts)

    result_payload["worker_backend"] = worker_backend
    result_payload["unit"] = "meter"
    result_payload["source_unit_scale_to_meters"] = prepared_ifc.unit_scale_to_meters
    result_payload["artifacts"] = artifacts
    result_payload["summary"] = _build_result_summary(result_payload["entities"], artifacts)
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


def _run_python_worker(request_payload: dict[str, Any]) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []

    for entity in request_payload["entities"]:
        result_entity = {
            "object_name": entity["object_name"],
            "global_id": entity.get("global_id"),
            "express_id": entity["express_id"],
            "name": entity.get("name"),
            "entity_type": entity["entity_type"],
            "has_representation": entity["has_representation"],
        }

        if entity.get("skip_reason"):
            result_entity.update(
                {
                    "mesh": None,
                    "vertex_count": 0,
                    "face_count": 0,
                    "component_count": 0,
                    "components": [],
                    "repair_actions": [],
                    "closed": False,
                    "manifold": False,
                    "outward_normals": False,
                    "volume_m3": 0.0,
                    "valid": False,
                    "reason": entity["skip_reason"],
                    "artifacts": {},
                }
            )
            entities.append(result_entity)
            continue

        normalized = normalize_mesh(
            entity["mesh"]["vertices"],
            entity["mesh"]["faces"],
        )
        normalized["artifacts"] = {}
        result_entity.update(normalized)
        entities.append(result_entity)

    return {
        "job_id": request_payload["job_id"],
        "schema": request_payload["schema"],
        "entities": entities,
    }


def _invoke_cpp_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
    completed = subprocess.run(
        [str(worker_binary), str(request_path), str(result_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "geometry worker failed"
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
        entity_artifact_key="raw_obj",
        directory_prefix="geometry/raw",
        spaces_all_relative=artifacts["raw_spaces_all"],
        openings_all_relative=artifacts["raw_openings"],
        per_space_list=artifacts["per_raw_space_objs"],
        per_opening_list=artifacts["per_raw_opening_objs"],
    )
    normalized_entity_artifacts = _write_entity_geometry(
        job_dir,
        result_entities,
        entity_artifact_key="normalized_obj",
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
    entity_artifact_key: str,
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


def _build_result_summary(entities: list[dict[str, Any]], artifacts: dict[str, Any]) -> dict[str, Any]:
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
