from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from .external_shell import ExternalShellResult, run_external_shell_classification
from .geometry_worker import GeometryPreprocessingResult, run_geometry_preprocessing
from .ifc_editing import (
    InvalidSpaceRemovalRequestError,
    InvalidSpaceResolutionRequestError,
    derive_ifc_resolving_space_clashes,
    derive_ifc_without_spaces,
)
from .ifc_extractor import ParsedIFC, PreparedIFC, build_extraction_report, parse_ifc_file, prepare_extraction
from .internal_boundaries import InternalBoundaryResult, run_internal_boundary_generation
from .models import JobState
from .preflight import PreflightValidationResult, run_preflight_validation
from .viewer_manifest import build_viewer_manifest


TERMINAL_STATES = {"complete", "failed"}


@dataclass(slots=True)
class JobProcessingBundle:
    prepared_ifc: PreparedIFC
    preprocessing: GeometryPreprocessingResult
    preflight: PreflightValidationResult
    internal_boundaries: InternalBoundaryResult
    external_shell: ExternalShellResult


class JobNotFoundError(FileNotFoundError):
    pass


class ArtifactNotFoundError(FileNotFoundError):
    pass


class InvalidJobOperationError(RuntimeError):
    pass


class JobService:
    def __init__(
        self,
        jobs_root: Path,
        stage_delay_seconds: float = 0.2,
        geometry_worker_binary: Path | None = None,
        exact_repair_mode: str = "preferred",
        exact_repair_worker_binary: Path | None = None,
        shell_worker_binary: Path | None = None,
        internal_boundary_thickness_threshold_m: float = 0.30,
        preflight_clash_tolerance_m: float = 0.01,
    ) -> None:
        self.jobs_root = jobs_root
        self.stage_delay_seconds = stage_delay_seconds
        resolved_exact_repair_binary = exact_repair_worker_binary or geometry_worker_binary
        self.geometry_worker_binary = resolved_exact_repair_binary
        self.exact_repair_mode = exact_repair_mode
        self.exact_repair_worker_binary = resolved_exact_repair_binary
        self.shell_worker_binary = shell_worker_binary
        self.internal_boundary_thickness_threshold_m = internal_boundary_thickness_threshold_m
        self.preflight_clash_tolerance_m = preflight_clash_tolerance_m
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._io_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="ifc-job-worker",
            daemon=True,
        )

    def start(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.recover_interrupted_jobs()
        if not self._worker.is_alive():
            self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def create_job(self, upload: UploadFile, *, external_shell_mode: str = "alpha_wrap") -> dict[str, Any]:
        job_id = str(uuid4())
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)

        input_path = job_dir / "input.ifc"
        with input_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        created_at = _utcnow()
        original_name = upload.filename or "upload.ifc"
        input_size = input_path.stat().st_size

        self._append_log(job_id, f"Upload received: {original_name} ({input_size} bytes)")
        debug_payload = self._build_debug_payload(
            job_id,
            created_at=created_at,
            original_name=original_name,
            input_size=input_size,
            external_shell_mode=external_shell_mode,
            history_message="Upload received",
        )
        self._write_debug(job_id, debug_payload)
        self._queue.put(job_id)

        return {
            "job_id": job_id,
            "state": "uploaded",
            "created_at": created_at,
            "status_url": f"/jobs/{job_id}",
            "artifacts_url": f"/jobs/{job_id}/artifacts",
        }

    def create_remove_spaces_rerun(
        self,
        job_id: str,
        *,
        space_global_ids: list[str] | None = None,
        space_express_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        parent_debug = self._load_debug(job_id)
        if parent_debug["state"] not in TERMINAL_STATES:
            raise InvalidJobOperationError("Reruns are only available after a job reaches complete or failed.")

        base_input_path = self._job_dir(job_id) / "input.ifc"
        if not base_input_path.exists():
            raise InvalidJobOperationError(f"Job {job_id} does not have an input.ifc artifact to derive from.")

        child_job_id = str(uuid4())
        child_job_dir = self._job_dir(child_job_id)
        child_job_dir.mkdir(parents=True, exist_ok=False)

        parent_derivation = parent_debug.get("derivation") or {}
        root_job_id = parent_derivation.get("root_job_id") or job_id
        external_shell_mode = (
            parent_debug.get("options", {}).get("external_shell_mode")
            or parent_debug.get("external_shell_preview", {}).get("mode_requested")
            or "alpha_wrap"
        )
        input_path = child_job_dir / "input.ifc"
        try:
            edit_result = derive_ifc_without_spaces(
                base_input_path,
                input_path,
                space_global_ids=space_global_ids,
                space_express_ids=space_express_ids,
            )
        except Exception:
            shutil.rmtree(child_job_dir, ignore_errors=True)
            raise

        derivation_payload = {
            "parent_job_id": job_id,
            "root_job_id": root_job_id,
            "operation": "remove_spaces",
            "removed_space_count": len(edit_result.removed_spaces),
            "removed_spaces": edit_result.removed_spaces,
        }
        edit_payload = {
            **derivation_payload,
            "requested_space_global_ids": edit_result.requested_space_global_ids,
            "requested_space_express_ids": edit_result.requested_space_express_ids,
            "remaining_space_count": edit_result.remaining_space_count,
        }
        self._write_json_file(child_job_dir / "edits" / "remove_spaces.json", edit_payload)

        created_at = _utcnow()
        original_name = parent_debug.get("input", {}).get("original_filename") or base_input_path.name
        input_size = input_path.stat().st_size
        debug_payload = self._build_debug_payload(
            child_job_id,
            created_at=created_at,
            original_name=original_name,
            input_size=input_size,
            external_shell_mode=external_shell_mode,
            history_message=(
                f"Derived from {job_id} after removing {len(edit_result.removed_spaces)} spaces"
            ),
            derivation=derivation_payload,
        )
        self._write_debug(child_job_id, debug_payload)
        self._append_log(
            child_job_id,
            (
                f"Derived job created from {job_id}; removed {len(edit_result.removed_spaces)} spaces "
                f"and wrote edits/remove_spaces.json"
            ),
        )
        self._queue.put(child_job_id)

        return {
            "job_id": child_job_id,
            "state": "uploaded",
            "created_at": created_at,
            "status_url": f"/jobs/{child_job_id}",
            "artifacts_url": f"/jobs/{child_job_id}/artifacts",
            "parent_job_id": job_id,
            "root_job_id": root_job_id,
            "removed_space_count": len(edit_result.removed_spaces),
        }

    def create_resolve_space_clashes_rerun(
        self,
        job_id: str,
        *,
        group_resolutions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        parent_debug = self._load_debug(job_id)
        if parent_debug["state"] not in TERMINAL_STATES:
            raise InvalidJobOperationError("Reruns are only available after a job reaches complete or failed.")

        base_input_path = self._job_dir(job_id) / "input.ifc"
        if not base_input_path.exists():
            raise InvalidJobOperationError(f"Job {job_id} does not have an input.ifc artifact to derive from.")

        parent_output = self._load_output_payload(job_id)
        preflight = parent_output.get("preflight") or {}
        clash_groups = {
            str(clash_group["clash_group_id"]): clash_group
            for clash_group in preflight.get("clash_groups", [])
            if clash_group.get("clash_group_id")
        }
        if not clash_groups:
            raise InvalidJobOperationError(f"Job {job_id} does not have any preflight clash groups to resolve.")

        validated_group_resolutions = self._validate_group_resolutions(group_resolutions or [], clash_groups)
        child_job_id = str(uuid4())
        child_job_dir = self._job_dir(child_job_id)
        child_job_dir.mkdir(parents=True, exist_ok=False)

        parent_derivation = parent_debug.get("derivation") or {}
        root_job_id = parent_derivation.get("root_job_id") or job_id
        external_shell_mode = (
            parent_debug.get("options", {}).get("external_shell_mode")
            or parent_debug.get("external_shell_preview", {}).get("mode_requested")
            or "alpha_wrap"
        )
        input_path = child_job_dir / "input.ifc"
        try:
            edit_result = derive_ifc_resolving_space_clashes(
                base_input_path,
                input_path,
                group_resolutions=validated_group_resolutions,
            )
        except Exception:
            shutil.rmtree(child_job_dir, ignore_errors=True)
            raise

        resolved_group_ids = [
            str(group_resolution["clash_group_id"])
            for group_resolution in edit_result.requested_group_resolutions
        ]
        derivation_payload = {
            "parent_job_id": job_id,
            "root_job_id": root_job_id,
            "operation": "resolve_space_clashes",
            "removed_space_count": len(edit_result.removed_spaces),
            "removed_spaces": edit_result.removed_spaces,
            "resolved_clash_group_count": len(resolved_group_ids),
            "resolved_clash_group_ids": resolved_group_ids,
        }
        edit_payload = {
            **derivation_payload,
            "requested_group_resolutions": edit_result.requested_group_resolutions,
            "requested_space_global_ids": edit_result.requested_space_global_ids,
            "requested_space_express_ids": edit_result.requested_space_express_ids,
            "remaining_space_count": edit_result.remaining_space_count,
        }
        self._write_json_file(child_job_dir / "edits" / "resolve_space_clashes.json", edit_payload)

        created_at = _utcnow()
        original_name = parent_debug.get("input", {}).get("original_filename") or base_input_path.name
        input_size = input_path.stat().st_size
        debug_payload = self._build_debug_payload(
            child_job_id,
            created_at=created_at,
            original_name=original_name,
            input_size=input_size,
            external_shell_mode=external_shell_mode,
            history_message=(
                f"Derived from {job_id} after resolving {len(resolved_group_ids)} clash groups"
            ),
            derivation=derivation_payload,
        )
        self._write_debug(child_job_id, debug_payload)
        self._append_log(
            child_job_id,
            (
                f"Derived job created from {job_id}; reviewed {len(resolved_group_ids)} clash groups, "
                f"removed {len(edit_result.removed_spaces)} spaces, and wrote edits/resolve_space_clashes.json"
            ),
        )
        self._queue.put(child_job_id)

        return {
            "job_id": child_job_id,
            "state": "uploaded",
            "created_at": created_at,
            "status_url": f"/jobs/{child_job_id}",
            "artifacts_url": f"/jobs/{child_job_id}/artifacts",
            "parent_job_id": job_id,
            "root_job_id": root_job_id,
            "removed_space_count": len(edit_result.removed_spaces),
            "resolved_clash_group_count": len(resolved_group_ids),
        }

    def get_status(self, job_id: str) -> dict[str, Any]:
        debug_data = self._load_debug(job_id)
        artifacts = list(self._artifact_snapshot(self._job_dir(job_id)).values())
        return {
            "job_id": debug_data["job_id"],
            "state": debug_data["state"],
            "created_at": debug_data["created_at"],
            "updated_at": debug_data["updated_at"],
            "error": debug_data.get("error"),
            "history": debug_data.get("history", []),
            "artifacts": artifacts,
            "derivation": debug_data.get("derivation"),
        }

    def list_artifacts(self, job_id: str) -> dict[str, Any]:
        debug_data = self._load_debug(job_id)
        artifacts = list(self._artifact_snapshot(self._job_dir(job_id)).values())
        return {
            "job_id": job_id,
            "state": debug_data["state"],
            "artifacts": artifacts,
        }

    def get_artifact_path(self, job_id: str, artifact_path: str) -> Path:
        job_dir = self._job_dir(job_id)
        if not job_dir.exists():
            raise JobNotFoundError(f"Job {job_id} does not exist")

        relative_path = Path(artifact_path.strip("/"))
        if not artifact_path or relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
            raise ArtifactNotFoundError(f"Unknown artifact: {artifact_path}")

        resolved_job_dir = job_dir.resolve()
        resolved_artifact_path = (job_dir / relative_path).resolve()
        try:
            resolved_artifact_path.relative_to(resolved_job_dir)
        except ValueError as exc:
            raise ArtifactNotFoundError(f"Unknown artifact: {artifact_path}") from exc

        if not resolved_artifact_path.exists() or not resolved_artifact_path.is_file():
            raise ArtifactNotFoundError(f"Artifact {artifact_path} is not available for job {job_id}")

        return resolved_artifact_path

    def recover_interrupted_jobs(self) -> None:
        for job_dir in sorted(self.jobs_root.iterdir()):
            if not job_dir.is_dir():
                continue

            debug_path = job_dir / "debug.json"
            if not debug_path.exists():
                continue

            try:
                debug_data = self._load_debug(job_dir.name)
            except (json.JSONDecodeError, JobNotFoundError):
                continue

            if debug_data["state"] in TERMINAL_STATES:
                continue

            self._fail_job(job_dir.name, "interrupted during previous run", append_history=True)

    def _worker_loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if job_id is None:
                self._queue.task_done()
                break

            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        try:
            self._transition(job_id, "parsing", "Parsing IFC and detecting schema")
            self._sleep_between_stages()
            parsed_ifc = self._run_parsing(job_id)

            self._transition(job_id, "preprocessing", "Normalizing tessellated geometry")
            self._sleep_between_stages()
            bundle = self._run_preprocessing(job_id, parsed_ifc)

            self._transition(job_id, "preflight", "Validating normalized spaces for topology and clashes")
            self._sleep_between_stages()
            bundle = self._run_preflight(job_id, bundle)
            if bundle.preflight.payload["status"] == "failed":
                self._fail_preflight_job(job_id, bundle)
                return

            self._transition(job_id, "internal_boundary", "Detecting shared boundaries between adjacent spaces")
            self._sleep_between_stages()
            bundle = self._run_internal_boundary(job_id, bundle)

            self._transition(job_id, "external_shell", "Classifying space surfaces against the building shell")
            self._sleep_between_stages()
            bundle = self._run_external_shell(job_id, bundle)

            self._transition(job_id, "classifying", "Building extraction report")
            self._sleep_between_stages()
            self._run_classifying(job_id, bundle)

            self._transition(job_id, "complete", "Job complete")
            self._append_log(job_id, "Job finished successfully")
        except Exception as exc:
            self._fail_job(job_id, str(exc), append_history=True)

    def _run_parsing(self, job_id: str) -> ParsedIFC:
        input_path = self._job_dir(job_id) / "input.ifc"
        if not input_path.exists():
            raise FileNotFoundError("input.ifc is missing")

        input_size = input_path.stat().st_size
        if input_size <= 0:
            raise ValueError("Uploaded IFC is empty")

        parsed_ifc = parse_ifc_file(input_path)

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data.setdefault("input", {})
            debug_data["input"]["size_bytes"] = input_size
            debug_data["input"]["validated"] = True
            debug_data["input"]["schema"] = parsed_ifc.schema

        self._update_debug(job_id, mutator)
        self._append_log(job_id, f"Parsed IFC schema {parsed_ifc.schema} ({input_size} bytes)")
        return parsed_ifc

    def _run_preprocessing(self, job_id: str, parsed_ifc: ParsedIFC) -> JobProcessingBundle:
        prepared_ifc = prepare_extraction(parsed_ifc)
        preprocessing = run_geometry_preprocessing(
            job_id,
            self._job_dir(job_id),
            prepared_ifc,
            exact_repair_mode=self.exact_repair_mode,
            exact_repair_worker_binary=self.exact_repair_worker_binary,
        )

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["extraction_preview"] = {
                "schema": prepared_ifc.schema,
                "number_of_spaces": len(prepared_ifc.spaces),
                "number_of_openings": len(prepared_ifc.openings),
                "source_unit_scale_to_meters": prepared_ifc.unit_scale_to_meters,
            }
            debug_data["preprocessing_preview"] = {
                "worker_backend": preprocessing.result["worker_backend"],
                "unit": preprocessing.result["unit"],
                "summary": preprocessing.result["summary"],
                "repair": preprocessing.result["repair"],
                "artifacts": preprocessing.result["artifacts"],
            }

        self._update_debug(job_id, mutator)
        self._append_log(
            job_id,
            (
                "Prepared extraction targets and normalized geometry: "
                f"{len(prepared_ifc.spaces)} spaces, {len(prepared_ifc.openings)} openings, "
                f"backend={preprocessing.result['worker_backend']}, "
                f"repair_status={preprocessing.result['repair']['summary']['effective_mode']}"
            ),
        )
        self._append_log(job_id, "Wrote geometry request/result, summary, and OBJ debug artifacts")
        return JobProcessingBundle(
            prepared_ifc=prepared_ifc,
            preprocessing=preprocessing,
            preflight=PreflightValidationResult(payload={}),
            internal_boundaries=InternalBoundaryResult(payload={}),
            external_shell=ExternalShellResult(payload={}),
        )

    def _run_preflight(self, job_id: str, bundle: JobProcessingBundle) -> JobProcessingBundle:
        preflight = run_preflight_validation(
            job_id,
            self._job_dir(job_id),
            bundle.preprocessing.result,
            clash_tolerance_m=self.preflight_clash_tolerance_m,
        )

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["preflight_preview"] = {
                "status": preflight.payload["status"],
                "summary": preflight.payload["summary"],
                "artifacts": preflight.payload["artifacts"],
            }

        self._update_debug(job_id, mutator)
        self._append_log(
            job_id,
            (
                "Preflight validation finished: "
                f"status={preflight.payload['status']}, "
                f"{preflight.payload['summary']['blocker_count']} blockers, "
                f"{preflight.payload['summary']['warning_count']} warnings"
            ),
        )
        return JobProcessingBundle(
            prepared_ifc=bundle.prepared_ifc,
            preprocessing=bundle.preprocessing,
            preflight=preflight,
            internal_boundaries=bundle.internal_boundaries,
            external_shell=bundle.external_shell,
        )

    def _run_internal_boundary(self, job_id: str, bundle: JobProcessingBundle) -> JobProcessingBundle:
        internal_boundaries = run_internal_boundary_generation(
            job_id,
            self._job_dir(job_id),
            bundle.preprocessing.result,
            threshold_m=self.internal_boundary_thickness_threshold_m,
        )

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["internal_boundaries_preview"] = {
                "threshold_m": internal_boundaries.payload["threshold_m"],
                "summary": internal_boundaries.payload["summary"],
                "artifacts": internal_boundaries.payload["artifacts"],
            }

        self._update_debug(job_id, mutator)
        self._append_log(
            job_id,
            (
                "Generated internal boundary graph: "
                f"{internal_boundaries.payload['summary']['adjacent_pair_count']} adjacency pairs, "
                f"{internal_boundaries.payload['summary']['shared_surface_count']} shared surfaces"
            ),
        )
        return JobProcessingBundle(
            prepared_ifc=bundle.prepared_ifc,
            preprocessing=bundle.preprocessing,
            preflight=bundle.preflight,
            internal_boundaries=internal_boundaries,
            external_shell=bundle.external_shell,
        )

    def _run_external_shell(self, job_id: str, bundle: JobProcessingBundle) -> JobProcessingBundle:
        debug_data = self._load_debug(job_id)
        mode_requested = (
            debug_data.get("options", {}).get("external_shell_mode")
            or debug_data.get("external_shell_preview", {}).get("mode_requested")
            or "alpha_wrap"
        )
        external_shell = run_external_shell_classification(
            job_id,
            self._job_dir(job_id),
            bundle.preprocessing.result,
            bundle.internal_boundaries.payload,
            mode_requested=mode_requested,
            worker_binary=self.shell_worker_binary,
        )

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["external_shell_preview"] = {
                "mode_requested": external_shell.payload["mode_requested"],
                "mode_effective": external_shell.payload["mode_effective"],
                "fallback_reason": external_shell.payload.get("fallback_reason"),
                "summary": external_shell.payload["summary"],
                "artifacts": external_shell.payload["artifacts"],
            }

        self._update_debug(job_id, mutator)
        self._append_log(
            job_id,
            (
                "Classified space surfaces against shell: "
                f"mode={external_shell.payload['mode_effective']}, "
                f"{external_shell.payload['summary']['candidate_surface_count']} surfaces, "
                f"{external_shell.payload['summary']['unclassified_count']} unclassified"
            ),
        )
        return JobProcessingBundle(
            prepared_ifc=bundle.prepared_ifc,
            preprocessing=bundle.preprocessing,
            preflight=bundle.preflight,
            internal_boundaries=bundle.internal_boundaries,
            external_shell=external_shell,
        )

    def _run_classifying(self, job_id: str, bundle: JobProcessingBundle) -> None:
        output_payload = build_extraction_report(
            job_id,
            bundle.prepared_ifc,
            bundle.preprocessing.result,
            bundle.preflight.payload,
            bundle.internal_boundaries.payload,
            bundle.external_shell.payload,
            derivation_info=self._load_debug(job_id).get("derivation"),
        )
        self._write_json_file(self._job_dir(job_id) / "output.json", output_payload)
        viewer_manifest = build_viewer_manifest(job_id, output_payload)
        self._write_json_file(self._job_dir(job_id) / "geometry" / "viewer_manifest.json", viewer_manifest)
        self._update_debug(job_id, lambda debug_data: debug_data.update({"result": output_payload}))
        self._append_log(
            job_id,
            (
                "Wrote output.json extraction report and viewer manifest "
                f"({output_payload['summary']['number_of_spaces']} spaces, "
                f"{output_payload['summary']['number_of_openings']} openings, "
                f"{output_payload['preprocessing']['summary']['valid_entities']} valid normalized entities, "
                f"{output_payload['preflight']['summary']['blocker_count']} preflight blockers, "
                f"{output_payload['internal_boundaries']['summary']['adjacent_pair_count']} adjacency pairs, "
                f"{output_payload['external_shell']['summary']['candidate_surface_count']} classified surfaces)"
            ),
        )

    def _fail_preflight_job(self, job_id: str, bundle: JobProcessingBundle) -> None:
        blocker_count = bundle.preflight.payload["summary"]["blocker_count"]
        first_message = bundle.preflight.payload["blockers"][0]["message"]
        error_message = (
            f"Preflight failed: {first_message}"
            if blocker_count == 1
            else f"Preflight failed: {first_message} (+{blocker_count - 1} more blockers)"
        )
        output_payload = build_extraction_report(
            job_id,
            bundle.prepared_ifc,
            bundle.preprocessing.result,
            bundle.preflight.payload,
            {},
            {},
            derivation_info=self._load_debug(job_id).get("derivation"),
            success=False,
            error=error_message,
        )
        viewer_manifest = build_viewer_manifest(job_id, output_payload)
        self._fail_job(
            job_id,
            error_message,
            append_history=True,
            output_payload=output_payload,
            viewer_manifest=viewer_manifest,
        )

    def _fail_job(
        self,
        job_id: str,
        error_message: str,
        append_history: bool,
        *,
        output_payload: dict[str, Any] | None = None,
        viewer_manifest: dict[str, Any] | None = None,
    ) -> None:
        failure_payload = output_payload or {
            "success": False,
            "job_id": job_id,
            "error": error_message,
        }
        self._write_json_file(self._job_dir(job_id) / "output.json", failure_payload)
        if viewer_manifest is not None:
            self._write_json_file(self._job_dir(job_id) / "geometry" / "viewer_manifest.json", viewer_manifest)
        self._append_log(job_id, f"Job failed: {error_message}")

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["state"] = "failed"
            debug_data["error"] = error_message
            debug_data["result"] = failure_payload
            if append_history:
                debug_data.setdefault("history", []).append(
                    {
                        "state": "failed",
                        "timestamp": _utcnow(),
                        "message": error_message,
                    }
                )

        self._update_debug(job_id, mutator)

    def _transition(self, job_id: str, state: JobState, message: str) -> None:
        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["state"] = state
            if state != "failed":
                debug_data["error"] = None
            debug_data.setdefault("history", []).append(
                {
                    "state": state,
                    "timestamp": _utcnow(),
                    "message": message,
                }
            )

        self._update_debug(job_id, mutator)
        self._append_log(job_id, message)

    def _update_debug(self, job_id: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        debug_path = self._debug_path(job_id)
        with self._io_lock:
            debug_data = self._read_json_file(debug_path)
            mutator(debug_data)
            debug_data["updated_at"] = _utcnow()
            self._write_debug_unlocked(job_id, debug_data)
            return debug_data

    def _write_debug(self, job_id: str, debug_data: dict[str, Any]) -> None:
        with self._io_lock:
            self._write_debug_unlocked(job_id, debug_data)

    def _write_debug_unlocked(self, job_id: str, debug_data: dict[str, Any]) -> None:
        debug_path = self._debug_path(job_id)
        debug_data["artifacts"] = self._artifact_snapshot(
            self._job_dir(job_id),
            extra_existing={"debug.json"},
        )
        self._write_json_atomically(debug_path, debug_data)

    def _load_debug(self, job_id: str) -> dict[str, Any]:
        debug_path = self._debug_path(job_id)
        if not debug_path.exists():
            raise JobNotFoundError(f"Job {job_id} does not exist")
        with self._io_lock:
            return self._read_json_file(debug_path)

    def _load_output_payload(self, job_id: str) -> dict[str, Any]:
        output_path = self._job_dir(job_id) / "output.json"
        if not output_path.exists():
            raise InvalidJobOperationError(f"Job {job_id} does not have an output.json artifact to derive from.")
        with self._io_lock:
            return self._read_json_file(output_path)

    @staticmethod
    def _validate_group_resolutions(
        group_resolutions: list[dict[str, Any]],
        clash_groups: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_group_resolutions: list[dict[str, Any]] = []
        seen_group_ids: set[str] = set()
        for group_resolution in group_resolutions:
            clash_group_id = str(group_resolution.get("clash_group_id") or "").strip()
            if not clash_group_id or clash_group_id in seen_group_ids:
                continue
            seen_group_ids.add(clash_group_id)

            clash_group = clash_groups.get(clash_group_id)
            if clash_group is None:
                raise InvalidSpaceResolutionRequestError(f"Unknown clash_group_id: {clash_group_id}")

            valid_member_global_ids = {
                str(space_ref["global_id"])
                for space_ref in clash_group.get("spaces", [])
                if space_ref.get("global_id")
            }
            valid_member_express_ids = {
                int(space_ref["express_id"])
                for space_ref in clash_group.get("spaces", [])
            }

            remove_space_global_ids = []
            for global_id in group_resolution.get("remove_space_global_ids", []):
                candidate = str(global_id).strip()
                if not candidate:
                    continue
                if candidate not in valid_member_global_ids:
                    raise InvalidSpaceResolutionRequestError(
                        f"Space {candidate} does not belong to clash group {clash_group_id}."
                    )
                if candidate not in remove_space_global_ids:
                    remove_space_global_ids.append(candidate)

            remove_space_express_ids = []
            for express_id in group_resolution.get("remove_space_express_ids", []):
                candidate = int(express_id)
                if candidate not in valid_member_express_ids:
                    raise InvalidSpaceResolutionRequestError(
                        f"Space #{candidate} does not belong to clash group {clash_group_id}."
                    )
                if candidate not in remove_space_express_ids:
                    remove_space_express_ids.append(candidate)

            if not remove_space_global_ids and not remove_space_express_ids:
                raise InvalidSpaceResolutionRequestError(
                    f"Select at least one IfcSpace to remove for clash group {clash_group_id}."
                )

            normalized_group_resolutions.append(
                {
                    "clash_group_id": clash_group_id,
                    "remove_space_global_ids": remove_space_global_ids,
                    "remove_space_express_ids": remove_space_express_ids,
                }
            )

        if not normalized_group_resolutions:
            raise InvalidSpaceResolutionRequestError("Select at least one clash group resolution before creating a rerun.")
        return normalized_group_resolutions

    def _artifact_snapshot(
        self,
        job_dir: Path,
        extra_existing: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        extra_existing = extra_existing or set()
        snapshot: dict[str, dict[str, Any]] = {}

        if job_dir.exists():
            for artifact_path in sorted(job_dir.rglob("*")):
                if not artifact_path.is_file():
                    continue
                if artifact_path.name.endswith(".tmp"):
                    continue
                relative_name = artifact_path.relative_to(job_dir).as_posix()
                try:
                    size_bytes = artifact_path.stat().st_size
                except FileNotFoundError:
                    continue
                snapshot[relative_name] = {
                    "name": relative_name,
                    "available": True,
                    "size_bytes": size_bytes,
                    "url": f"/jobs/{job_dir.name}/artifacts/{relative_name}",
                }

        for relative_name in sorted(extra_existing):
            if relative_name in snapshot:
                continue
            snapshot[relative_name] = {
                "name": relative_name,
                "available": True,
                "size_bytes": None,
                "url": f"/jobs/{job_dir.name}/artifacts/{relative_name}",
            }

        return snapshot

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _debug_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "debug.json"

    def _build_debug_payload(
        self,
        job_id: str,
        *,
        created_at: str,
        original_name: str,
        input_size: int,
        external_shell_mode: str,
        history_message: str,
        derivation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "state": "uploaded",
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
            "history": [
                {
                    "state": "uploaded",
                    "timestamp": created_at,
                    "message": history_message,
                }
            ],
            "input": {
                "original_filename": original_name,
                "size_bytes": input_size,
                "validated": False,
            },
            "options": {
                "external_shell_mode": external_shell_mode,
                "exact_repair_mode": self.exact_repair_mode,
            },
            "derivation": derivation,
            "preflight_preview": {
                "status": None,
                "summary": {},
                "artifacts": {},
            },
            "external_shell_preview": {
                "mode_requested": external_shell_mode,
                "mode_effective": None,
                "fallback_reason": None,
            },
        }

    def _append_log(self, job_id: str, message: str) -> None:
        log_line = f"[{_utcnow()}] {message}\n"
        log_path = self._job_dir(job_id) / "logs.txt"
        with self._io_lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(log_line)

    def _write_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        with self._io_lock:
            self._write_json_atomically(path, payload)

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _sleep_between_stages(self) -> None:
        if self.stage_delay_seconds > 0:
            time.sleep(self.stage_delay_seconds)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
