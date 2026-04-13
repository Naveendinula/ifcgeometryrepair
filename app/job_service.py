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

from .geometry_worker import GeometryPreprocessingResult, run_geometry_preprocessing
from .ifc_extractor import ParsedIFC, PreparedIFC, build_extraction_report, parse_ifc_file, prepare_extraction
from .models import JobState
from .viewer_manifest import build_viewer_manifest


TERMINAL_STATES = {"complete", "failed"}


@dataclass(slots=True)
class JobProcessingBundle:
    prepared_ifc: PreparedIFC
    preprocessing: GeometryPreprocessingResult


class JobNotFoundError(FileNotFoundError):
    pass


class ArtifactNotFoundError(FileNotFoundError):
    pass


class JobService:
    def __init__(
        self,
        jobs_root: Path,
        stage_delay_seconds: float = 0.2,
        geometry_worker_binary: Path | None = None,
    ) -> None:
        self.jobs_root = jobs_root
        self.stage_delay_seconds = stage_delay_seconds
        self.geometry_worker_binary = geometry_worker_binary
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

    def create_job(self, upload: UploadFile) -> dict[str, Any]:
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

        debug_payload = {
            "job_id": job_id,
            "state": "uploaded",
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
            "history": [
                {
                    "state": "uploaded",
                    "timestamp": created_at,
                    "message": "Upload received",
                }
            ],
            "input": {
                "original_filename": original_name,
                "size_bytes": input_size,
                "validated": False,
            },
        }
        self._write_debug(job_id, debug_payload)
        self._queue.put(job_id)

        return {
            "job_id": job_id,
            "state": "uploaded",
            "created_at": created_at,
            "status_url": f"/jobs/{job_id}",
            "artifacts_url": f"/jobs/{job_id}/artifacts",
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
            worker_binary=self.geometry_worker_binary,
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
                "artifacts": preprocessing.result["artifacts"],
            }

        self._update_debug(job_id, mutator)
        self._append_log(
            job_id,
            (
                "Prepared extraction targets and normalized geometry: "
                f"{len(prepared_ifc.spaces)} spaces, {len(prepared_ifc.openings)} openings, "
                f"backend={preprocessing.result['worker_backend']}"
            ),
        )
        self._append_log(job_id, "Wrote geometry request/result, summary, and OBJ debug artifacts")
        return JobProcessingBundle(prepared_ifc=prepared_ifc, preprocessing=preprocessing)

    def _run_classifying(self, job_id: str, bundle: JobProcessingBundle) -> None:
        output_payload = build_extraction_report(
            job_id,
            bundle.prepared_ifc,
            bundle.preprocessing.result,
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
                f"{output_payload['preprocessing']['summary']['valid_entities']} valid normalized entities)"
            ),
        )

    def _fail_job(self, job_id: str, error_message: str, append_history: bool) -> None:
        failure_payload = {
            "success": False,
            "job_id": job_id,
            "error": error_message,
        }
        self._write_json_file(self._job_dir(job_id) / "output.json", failure_payload)
        self._append_log(job_id, f"Job failed: {error_message}")

        def mutator(debug_data: dict[str, Any]) -> None:
            debug_data["state"] = "failed"
            debug_data["error"] = error_message
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
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _sleep_between_stages(self) -> None:
        if self.stage_delay_seconds > 0:
            time.sleep(self.stage_delay_seconds)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
