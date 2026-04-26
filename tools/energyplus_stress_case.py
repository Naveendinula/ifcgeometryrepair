from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ifcopenshell.api import run
from starlette.datastructures import UploadFile

from app.config import Settings
from app.job_service import JobService


DEFAULT_IFC_PATH = Path("ifc-file") / "energyplus_fault_stress.ifc"
DEFAULT_DEBUG_PATH = Path("ifc-file") / "energyplus_fault_stress_debug.json"
DEFAULT_JOBS_ROOT = Path("tmp") / "energyplus_stress_jobs"


@dataclass(frozen=True, slots=True)
class FaultExpectation:
    fault_id: str
    entity_names: tuple[str, ...]
    energyplus_risk: str
    expected_app_signal: str
    expected_stage: str


FAULT_EXPECTATIONS: tuple[FaultExpectation, ...] = (
    FaultExpectation(
        fault_id="exact_duplicate_spaces",
        entity_names=("EPLUS_ERROR_DUPLICATE_A", "EPLUS_ERROR_DUPLICATE_B"),
        energyplus_risk="Coincident zones/surfaces; EnergyPlus commonly reports duplicate or overlapping heat-transfer surfaces.",
        expected_app_signal='preflight.clash_groups[].classification == "exact_duplicate"',
        expected_stage="preflight",
    ),
    FaultExpectation(
        fault_id="partial_overlap_spaces",
        entity_names=("EPLUS_ERROR_PARTIAL_OVERLAP_A", "EPLUS_ERROR_PARTIAL_OVERLAP_B"),
        energyplus_risk="Partially overlapping zones; downstream surfaces cannot form a clean thermal zone enclosure.",
        expected_app_signal='preflight.clash_groups[].classification == "partial_overlap"',
        expected_stage="preflight",
    ),
    FaultExpectation(
        fault_id="contained_fragment_space",
        entity_names=("EPLUS_ERROR_CONTAINED_OUTER", "EPLUS_ERROR_CONTAINED_FRAGMENT"),
        energyplus_risk="Nested zone fragment; EnergyPlus export should not include a fully contained duplicate volume.",
        expected_app_signal='preflight.clash_groups[].classification == "contained_fragment"',
        expected_stage="preflight",
    ),
    FaultExpectation(
        fault_id="missing_space_representation",
        entity_names=("EPLUS_ERROR_MISSING_GEOMETRY_SPACE",),
        energyplus_risk="Zone has no closed solid to export; EnergyPlus cannot receive a valid zone enclosure.",
        expected_app_signal='preflight.blockers[].code == "invalid_space_solid"',
        expected_stage="preflight",
    ),
    FaultExpectation(
        fault_id="thin_sliver_space",
        entity_names=("EPLUS_WARNING_THIN_SLIVER_SPACE",),
        energyplus_risk="Very thin zone can lead to tiny/near-coincident surfaces and poor EnergyPlus geometry robustness.",
        expected_app_signal="coverage_gap: current preflight does not hard-fail thin-but-closed spaces",
        expected_stage="preflight_or_later_visual_validation",
    ),
    FaultExpectation(
        fault_id="floating_opening_candidate",
        entity_names=("EPLUS_WARNING_FLOATING_OPENING",),
        energyplus_risk="Subsurface/opening not hosted by a coplanar parent surface; EnergyPlus typically rejects or warns on invalid subsurfaces.",
        expected_app_signal="not_reached_when_preflight_blocks; should be checked with a separate gbXML-stage fixture",
        expected_stage="opening_integration_or_gbxml_preflight",
    ),
)


def build_energyplus_stress_ifc(output_path: Path) -> dict[str, Any]:
    model, body_context, building, storey = _base_model()

    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_CONTROL_VALID_ROOM",
        placement=(0.0, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_DUPLICATE_A",
        placement=(8.0, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_DUPLICATE_B",
        placement=(8.0, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_PARTIAL_OVERLAP_A",
        placement=(16.0, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_PARTIAL_OVERLAP_B",
        placement=(19.6, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_CONTAINED_OUTER",
        placement=(28.0, 0.0, 0.0),
        represented=True,
        footprint=(5000.0, 5000.0),
        height=3.0,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_CONTAINED_FRAGMENT",
        placement=(28.0, 0.0, 0.0),
        represented=True,
        footprint=(1000.0, 1000.0),
        height=1.5,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_ERROR_MISSING_GEOMETRY_SPACE",
        placement=(36.0, 0.0, 0.0),
        represented=False,
    )
    _add_space(
        model,
        body_context,
        storey,
        name="EPLUS_WARNING_THIN_SLIVER_SPACE",
        placement=(44.0, 0.0, 0.0),
        represented=True,
        footprint=(50.0, 3000.0),
        height=2.8,
    )
    _add_opening(
        model,
        body_context,
        storey,
        name="EPLUS_WARNING_FLOATING_OPENING",
        placement=(52.0, 0.0, 1.0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model.to_string(), encoding="utf-8")

    return {
        "ifc_path": str(output_path),
        "building_name": building.Name,
        "storey_name": storey.Name,
        "space_count": len(model.by_type("IfcSpace")),
        "opening_count": len(model.by_type("IfcOpeningElement")),
        "fault_expectations": [asdict(expectation) for expectation in FAULT_EXPECTATIONS],
    }


def run_energyplus_stress_debug(
    ifc_path: Path,
    debug_path: Path,
    *,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    external_shell_mode: str = "alpha_wrap",
    run_app: bool = True,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Portable IFC geometry stress case for checking whether the app catches common "
            "EnergyPlus/gBXML blocking geometry defects before clean export."
        ),
        "generated_ifc": build_energyplus_stress_ifc(ifc_path),
        "expected_terminal_behavior": (
            "This combined stress IFC is expected to fail at preflight. Later stages are intentionally "
            "not expected to run until the blocking spaces are removed or split into separate stage fixtures."
        ),
        "robustness_notes": [
            "Use this as a hard-failure smoke test, not as the only export-readiness test.",
            "A separate clean-control IFC should complete through gbXML preflight with zero blockers.",
            "A separate gbXML-stage fixture is needed for coplanar/non-coplanar opening validation because this file stops at preflight.",
            "Thin but closed spaces are included as a current coverage gap: the app may not fail them before visual/gbXML validation.",
        ],
    }

    if run_app:
        run_summary = _run_pipeline(ifc_path, jobs_root=jobs_root, external_shell_mode=external_shell_mode)
        manifest["app_run"] = run_summary
        manifest["checks"] = _evaluate_run(run_summary)
    else:
        manifest["app_run"] = None
        manifest["checks"] = []

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _run_pipeline(ifc_path: Path, *, jobs_root: Path, external_shell_mode: str) -> dict[str, Any]:
    settings = Settings()
    service = JobService(
        jobs_root=jobs_root,
        stage_delay_seconds=0.0,
        geometry_worker_binary=settings.geometry_worker_binary,
        exact_repair_mode=settings.exact_repair_mode,
        exact_repair_worker_binary=settings.exact_repair_worker_binary,
        shell_worker_binary=settings.shell_worker_binary,
    )
    service.start()
    try:
        with ifc_path.open("rb") as handle:
            upload = UploadFile(file=handle, filename=ifc_path.name)
            created = service.create_job(upload, external_shell_mode=external_shell_mode)

        job_id = created["job_id"]
        deadline = time.time() + 120.0
        status = service.get_status(job_id)
        while status["state"] not in {"complete", "failed"} and time.time() < deadline:
            time.sleep(0.2)
            status = service.get_status(job_id)

        job_dir = jobs_root / job_id
        output_payload = _read_json_if_exists(job_dir / "output.json")
        preflight_payload = _read_json_if_exists(job_dir / "geometry" / "preflight.json")
        gbxml_preflight_payload = _read_json_if_exists(job_dir / "geometry" / "gbxml_preflight.json")

        return {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "state": status.get("state"),
            "error": status.get("error"),
            "history_states": [entry["state"] for entry in status.get("history", [])],
            "artifact_names": [artifact["name"] for artifact in status.get("artifacts", []) if artifact.get("available")],
            "output_summary": output_payload.get("summary", {}),
            "preprocessing_summary": output_payload.get("preprocessing", {}).get("summary", {}),
            "preflight": _compact_preflight(preflight_payload or output_payload.get("preflight", {})),
            "external_shell_summary": output_payload.get("external_shell", {}).get("summary", {}),
            "opening_integration_summary": output_payload.get("opening_integration", {}).get("summary", {}),
            "gbxml_preflight": _compact_gbxml_preflight(gbxml_preflight_payload or output_payload.get("gbxml_preflight", {})),
        }
    finally:
        service.stop()


def _evaluate_run(run_summary: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = run_summary.get("preflight", {})
    clash_classes = {group.get("classification") for group in preflight.get("clash_groups", [])}
    blocker_codes = {blocker.get("code") for blocker in preflight.get("blockers", [])}
    history_states = run_summary.get("history_states", [])

    checks = [
        _check(
            "stops_at_preflight",
            run_summary.get("state") == "failed"
            and "preflight" in history_states
            and "internal_boundary" not in history_states,
            "Job should stop at preflight because this file intentionally contains hard blockers.",
            {"state": run_summary.get("state"), "history_states": history_states},
        ),
        _check(
            "detects_invalid_space_solid",
            "invalid_space_solid" in blocker_codes,
            "Missing IfcSpace geometry should produce an invalid_space_solid preflight blocker.",
            {"blocker_codes": sorted(code for code in blocker_codes if code)},
        ),
        _check(
            "detects_exact_duplicate",
            "exact_duplicate" in clash_classes,
            "Coincident spaces should be classified as exact_duplicate.",
            {"clash_classes": sorted(clash for clash in clash_classes if clash)},
        ),
        _check(
            "detects_partial_overlap",
            "partial_overlap" in clash_classes,
            "Partially intersecting spaces should be classified as partial_overlap.",
            {"clash_classes": sorted(clash for clash in clash_classes if clash)},
        ),
        _check(
            "detects_contained_fragment",
            "contained_fragment" in clash_classes,
            "Nested fragment spaces should be classified as contained_fragment.",
            {"clash_classes": sorted(clash for clash in clash_classes if clash)},
        ),
        _check(
            "manual_resolution_required",
            preflight.get("summary", {}).get("manual_resolution_group_count", 0) >= 1,
            "Partial overlaps should force manual review rather than automatic clean export.",
            {"summary": preflight.get("summary", {})},
        ),
        _check(
            "later_export_stages_not_run",
            not run_summary.get("external_shell_summary") and not run_summary.get("gbxml_preflight", {}).get("status"),
            "External shell, opening integration, and gbXML preflight should not run after hard preflight failure.",
            {
                "external_shell_summary": run_summary.get("external_shell_summary"),
                "gbxml_status": run_summary.get("gbxml_preflight", {}).get("status"),
            },
        ),
    ]
    return checks


def _compact_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "summary": payload.get("summary", {}),
        "blockers": [
            {
                "code": blocker.get("code"),
                "message": blocker.get("message"),
                "entity": blocker.get("entity"),
            }
            for blocker in payload.get("blockers", [])
        ],
        "warnings": [
            {
                "code": warning.get("code"),
                "message": warning.get("message"),
                "entity": warning.get("entity"),
            }
            for warning in payload.get("warnings", [])
        ],
        "clash_groups": [
            {
                "clash_group_id": group.get("clash_group_id"),
                "classification": group.get("classification"),
                "resolution_status": group.get("resolution_status"),
                "recommended_resolution": group.get("recommended_resolution"),
                "spaces": group.get("spaces", []),
                "pair_count": len(group.get("pairs", [])),
            }
            for group in payload.get("clash_groups", [])
        ],
    }


def _compact_gbxml_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "summary": payload.get("summary", {}),
        "blockers": [
            {"code": blocker.get("code"), "message": blocker.get("message")}
            for blocker in payload.get("blockers", [])
        ],
        "warnings": [
            {"code": warning.get("code"), "message": warning.get("message")}
            for warning in payload.get("warnings", [])
        ],
    }


def _check(check_id: str, passed: bool, expectation: str, observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "pass" if passed else "fail",
        "expectation": expectation,
        "observed": observed,
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _add_space(
    model: Any,
    body_context: Any,
    storey: Any,
    *,
    name: str,
    placement: tuple[float, float, float],
    represented: bool,
    footprint: tuple[float, float] = (4000.0, 3000.0),
    height: float = 2.8,
) -> Any:
    space = run("root.create_entity", model, ifc_class="IfcSpace", name=name)
    run("aggregate.assign_object", model, products=[space], relating_object=storey)
    run("geometry.edit_object_placement", model, product=space, matrix=_placement_matrix(*placement))

    if represented:
        profile = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=footprint[0], YDim=footprint[1])
        representation = run("geometry.add_profile_representation", model, context=body_context, profile=profile, depth=height)
        run("geometry.assign_representation", model, product=space, representation=representation)

    return space


def _add_opening(
    model: Any,
    body_context: Any,
    storey: Any,
    *,
    name: str,
    placement: tuple[float, float, float],
) -> Any:
    opening = run("root.create_entity", model, ifc_class="IfcOpeningElement", name=name)
    run("spatial.assign_container", model, products=[opening], relating_structure=storey)
    run("geometry.edit_object_placement", model, product=opening, matrix=_placement_matrix(*placement))
    profile = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=1000.0, YDim=250.0)
    representation = run("geometry.add_profile_representation", model, context=body_context, profile=profile, depth=2.0)
    run("geometry.assign_representation", model, product=opening, representation=representation)
    return opening


def _base_model() -> tuple[Any, Any, Any, Any]:
    model = run("project.create_file", version="IFC4")
    project = run("root.create_entity", model, ifc_class="IfcProject", name="EnergyPlus Geometry Stress Project")
    run("unit.assign_unit", model)
    model_context = run("context.add_context", model, context_type="Model")
    body_context = run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_context,
    )

    site = run("root.create_entity", model, ifc_class="IfcSite", name="Stress Site")
    building = run("root.create_entity", model, ifc_class="IfcBuilding", name="EnergyPlus Stress Building")
    storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Stress Level 1")

    run("aggregate.assign_object", model, products=[site], relating_object=project)
    run("aggregate.assign_object", model, products=[building], relating_object=site)
    run("aggregate.assign_object", model, products=[storey], relating_object=building)

    run("geometry.edit_object_placement", model, product=site)
    run("geometry.edit_object_placement", model, product=building)
    run("geometry.edit_object_placement", model, product=storey)

    return model, body_context, building, storey


def _placement_matrix(x: float, y: float, z: float) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run an EnergyPlus/gBXML geometry stress IFC.")
    parser.add_argument("--ifc-out", type=Path, default=DEFAULT_IFC_PATH)
    parser.add_argument("--debug-out", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    parser.add_argument("--skip-run", action="store_true", help="Only create the IFC and expected manifest.")
    parser.add_argument("--external-shell-mode", choices=("alpha_wrap", "heuristic"), default="alpha_wrap")
    args = parser.parse_args()

    manifest = run_energyplus_stress_debug(
        args.ifc_out,
        args.debug_out,
        jobs_root=args.jobs_root,
        external_shell_mode=args.external_shell_mode,
        run_app=not args.skip_run,
    )
    failed_checks = [check for check in manifest.get("checks", []) if check.get("status") != "pass"]
    print(json.dumps({"ifc": str(args.ifc_out), "debug": str(args.debug_out), "failed_checks": failed_checks}, indent=2))


if __name__ == "__main__":
    main()
