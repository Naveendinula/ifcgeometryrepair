from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

import ifcopenshell
import pytest
from fastapi.testclient import TestClient

import app.external_shell as external_shell
import app.geometry_worker as geometry_worker
from app.config import Settings
from app.main import create_app
from app.mesh_normalizer import normalize_mesh
from tests.ifc_factory import (
    build_contained_fragment_fixture,
    build_duplicate_rooms_fixture,
    build_corridor_room_fixture,
    build_empty_fixture,
    build_extraction_fixture,
    build_overlapping_rooms_fixture,
    build_separated_rooms_fixture,
    build_shared_wall_fixture,
    build_single_room_fixture,
)


def wait_for_terminal_state(client: TestClient, job_id: str, timeout: float = 5.0) -> tuple[dict, list[str]]:
    deadline = time.time() + timeout
    seen_states: list[str] = []

    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        seen_states.append(payload["state"])
        if payload["state"] in {"complete", "failed"}:
            return payload, seen_states
        time.sleep(0.02)

    raise AssertionError(f"Timed out waiting for terminal state for job {job_id}")


def make_test_root(label: str) -> Path:
    root = Path("jobs") / "test_runs" / f"{label}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def build_client(
    jobs_root: Path,
    *,
    exact_repair_mode: str = "preferred",
    exact_repair_worker_binary: Path | None = None,
    shell_worker_binary: Path | None = None,
) -> TestClient:
    resolved_shell_worker_binary = shell_worker_binary or make_fake_shell_worker_binary(
        jobs_root / "fake-shell-worker.py"
    )
    app = create_app(
        Settings(
            jobs_root=jobs_root,
            stage_delay_seconds=0.02,
            exact_repair_mode=exact_repair_mode,
            exact_repair_worker_binary=exact_repair_worker_binary or jobs_root / "missing-worker.exe",
            shell_worker_binary=resolved_shell_worker_binary,
            internal_boundary_thickness_threshold_m=0.30,
            alpha_wrap_alpha_m=1.0,
            alpha_wrap_offset_m=0.01,
            preflight_clash_tolerance_m=0.01,
        )
    )
    return TestClient(app)


def wait_for_output_report(client: TestClient, job_id: str) -> dict:
    response = client.get(f"/jobs/{job_id}/artifacts/output.json")
    assert response.status_code == 200
    return response.json()


def make_fake_worker_binary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def write_shell_worker_script(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.strip(), encoding="utf-8")
    return path


def make_fake_shell_worker_binary(path: Path) -> Path:
    return write_shell_worker_script(
        path,
        """
import json
import sys


def build_box_mesh(vertices):
    min_x = min(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    min_z = min(vertex[2] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    max_y = max(vertex[1] for vertex in vertices)
    max_z = max(vertex[2] for vertex in vertices)
    return {
        "vertices": [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        "faces": [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
    }


request_path = sys.argv[1]
result_path = sys.argv[2]
request_payload = json.loads(open(request_path, encoding="utf-8").read())
vertices = []
for space in request_payload.get("space_meshes", []):
    mesh = space.get("mesh") or {}
    vertices.extend(mesh.get("vertices", []))
shell_mesh = {"vertices": [], "faces": []} if not vertices else build_box_mesh(vertices)
payload = {
    "contract_version": 1,
    "status": "ok",
    "backend": "cpp-cgal-alpha-wrap",
    "alpha_m_effective": request_payload.get("alpha_m_effective"),
    "offset_m_effective": request_payload.get("offset_m_effective"),
    "generation_time_ms": 1.0,
    "shell_mesh": shell_mesh,
}
open(result_path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
""".strip(),
    )


def build_exact_repair_response(request_payload: dict[str, Any]) -> dict[str, Any]:
    spaces = []
    for space in request_payload.get("spaces", []):
        normalized = normalize_mesh(space["mesh"]["vertices"], space["mesh"]["faces"])
        normalized.update(
            {
                "global_id": space.get("global_id"),
                "express_id": space["express_id"],
                "name": space.get("name"),
                "repair_backend": "cpp-cgal",
                "repair_status": "exact_passthrough" if normalized["valid"] else "exact_repaired",
                "repair_reason": None,
            }
        )
        spaces.append(normalized)
    return {
        "contract_version": geometry_worker.EXACT_REPAIR_CONTRACT_VERSION,
        "status": "ok",
        "worker_backend": "cpp-cgal",
        "reason": None,
        "spaces": spaces,
    }


def test_simple_room_preprocesses_to_valid_positive_solid() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("single-room")

    try:
        with build_client(jobs_root) as client:
            response = client.post(
                "/jobs",
                files={"file": ("simple-room.ifc", fixture.content, "application/octet-stream")},
            )
            assert response.status_code == 202

            created = response.json()
            job_id = created["job_id"]
            job_dir = jobs_root / job_id

            assert job_dir.exists()
            assert (job_dir / "input.ifc").exists()
            assert (job_dir / "logs.txt").exists()
            assert (job_dir / "debug.json").exists()

            status_payload, seen_states = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"
            assert [entry["state"] for entry in status_payload["history"]] == [
                "uploaded",
                "parsing",
                "preprocessing",
                "preflight",
                "internal_boundary",
                "external_candidates",
                "external_shell",
                "classifying",
                "complete",
            ]
            assert len(set(seen_states)) >= 2

            artifacts_response = client.get(f"/jobs/{job_id}/artifacts")
            assert artifacts_response.status_code == 200
            artifact_names = {item["name"] for item in artifacts_response.json()["artifacts"]}
            assert {
                "logs.txt",
                "debug.json",
                "output.json",
                "preview.obj",
                "geometry/request.json",
                "geometry/result.json",
                "geometry/geometry_summary.json",
                "geometry/preflight.json",
                "geometry/internal_boundaries.json",
                "geometry/internal_boundaries.obj",
                "geometry/external_candidates/result.json",
                "geometry/external_candidates/candidates_all.obj",
                "geometry/external_shell/request.json",
                "geometry/external_shell/result.json",
                "geometry/external_shell/shell.obj",
                "geometry/external_shell/surfaces_all.obj",
                "geometry/external_shell/classes/external_wall.obj",
                "geometry/external_shell/classes/roof.obj",
                "geometry/external_shell/classes/ground_floor.obj",
                "geometry/external_shell/classes/internal_void.obj",
                "geometry/external_shell/classes/unclassified.obj",
                "geometry/viewer_manifest.json",
                "geometry/raw/spaces_all.obj",
                "geometry/raw/openings.obj",
                "geometry/spaces_all.obj",
                "geometry/openings.obj",
            }.issubset(artifact_names)

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["success"] is True
            assert output_payload["schema"] == "IFC4"
            assert output_payload["summary"]["number_of_spaces"] == fixture.expected_space_count
            assert output_payload["summary"]["number_of_openings"] == fixture.expected_opening_count
            assert output_payload["preprocessing"]["worker_backend"] == "python"
            assert output_payload["preflight"]["status"] == "passed"
            assert output_payload["preflight"]["summary"]["blocker_count"] == 0
            assert output_payload["preprocessing"]["summary"]["valid_entities"] == 1
            assert output_payload["preprocessing"]["artifacts"]["viewer_manifest"] == "geometry/viewer_manifest.json"
            assert output_payload["preprocessing"]["artifacts"]["raw_spaces_all"] == "geometry/raw/spaces_all.obj"
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 0
            assert output_payload["internal_boundaries"]["artifacts"]["detail"] == "geometry/internal_boundaries.json"
            assert output_payload["external_candidates"]["summary"]["candidate_surface_count"] == 6
            assert output_payload["external_candidates"]["artifacts"]["result"] == "geometry/external_candidates/result.json"
            assert output_payload["external_shell"]["summary"]["candidate_surface_count"] == 6
            assert output_payload["external_shell"]["summary"]["per_class_counts"]["external_wall"] == 4
            assert output_payload["external_shell"]["summary"]["per_class_counts"]["roof"] == 1
            assert output_payload["external_shell"]["summary"]["per_class_counts"]["ground_floor"] == 1
            assert output_payload["external_shell"]["artifacts"]["result"] == "geometry/external_shell/result.json"

            room = output_payload["spaces"][0]
            assert room["name"] == fixture.represented_space_name
            assert room["geometry_ok"] is True
            assert room["closed"] is True
            assert room["outward_normals"] is True
            assert room["volume_m3"] > 0
            assert room["storey"]["name"] == fixture.storey_name
            assert room["building"]["name"] == fixture.building_name
            assert "obj" in room["artifacts"]
            assert room["artifacts"]["normalized_obj"]
            assert room["artifacts"]["raw_obj"]
            assert (job_dir / room["artifacts"]["obj"]).exists()

            geometry_summary = json.loads((job_dir / "geometry" / "geometry_summary.json").read_text(encoding="utf-8"))
            assert geometry_summary["summary"]["valid_entities"] == 1
            assert geometry_summary["entities"][0]["closed"] is True

            internal_boundaries = json.loads((job_dir / "geometry" / "internal_boundaries.json").read_text(encoding="utf-8"))
            assert internal_boundaries["summary"]["shared_surface_count"] == 0
            assert internal_boundaries["adjacencies"] == []

            external_candidates_payload = json.loads((job_dir / "geometry" / "external_candidates" / "result.json").read_text(encoding="utf-8"))
            assert external_candidates_payload["summary"]["candidate_surface_count"] == 6
            assert len(external_candidates_payload["spaces"][0]["candidate_surface_ids"]) == 6

            external_shell_payload = json.loads((job_dir / "geometry" / "external_shell" / "result.json").read_text(encoding="utf-8"))
            assert external_shell_payload["mode_requested"] == "alpha_wrap"
            assert external_shell_payload["mode_effective"] == "alpha_wrap"
            assert external_shell_payload["fallback_reason"] is None
            assert output_payload["external_shell"]["shell_backend"] == "cpp-cgal-alpha-wrap"
            assert external_shell_payload["shell_backend"] == "cpp-cgal-alpha-wrap"
            assert external_shell_payload["alpha_wrap"]["status"] == "ok"
            assert external_shell_payload["alpha_wrap"]["backend"] == "cpp-cgal-alpha-wrap"
            assert external_shell_payload["alpha_wrap"]["alpha_m_effective"] == pytest.approx(1.0)
            assert external_shell_payload["alpha_wrap"]["offset_m_effective"] == pytest.approx(0.01)
            assert external_shell_payload["summary"]["candidate_surface_count"] == 6

            viewer_manifest = json.loads((job_dir / "geometry" / "viewer_manifest.json").read_text(encoding="utf-8"))
            assert viewer_manifest["summary"]["space_count"] == 1
            assert viewer_manifest["layers"]["raw_ifc_preview"]["available"] is True
            assert viewer_manifest["layers"]["normalized_spaces"]["available"] is True
            assert viewer_manifest["layers"]["failed_entities"]["available"] is False
            assert viewer_manifest["layers"]["envelope_shell"]["available"] is True
            assert viewer_manifest["layers"]["surface_classification"]["available"] is True
            assert viewer_manifest["entities"][0]["global_id"] == room["global_id"]
            assert len(viewer_manifest["surface_entities"]) == 6

            raw_spaces_text = (job_dir / "geometry" / "raw" / "spaces_all.obj").read_text(encoding="utf-8")
            normalized_spaces_text = (job_dir / "geometry" / "spaces_all.obj").read_text(encoding="utf-8")
            classified_surfaces_text = (job_dir / "geometry" / "external_shell" / "surfaces_all.obj").read_text(encoding="utf-8")
            assert f"o {room['global_id']}" in raw_spaces_text
            assert f"o {room['global_id']}" in normalized_spaces_text
            assert "o ec_0_0" in classified_surfaces_text
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_missing_exact_repair_worker_falls_back_to_python_and_records_report() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("exact-repair-missing")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("single-room.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = wait_for_output_report(client, job_id)
            room = output_payload["spaces"][0]
            repair_payload = output_payload["preprocessing"]["repair"]

            assert output_payload["preprocessing"]["worker_backend"] == "python"
            assert output_payload["preprocessing"]["artifacts"]["repair_report"] == "geometry/repair_report.json"
            assert repair_payload["effective_mode"] == "python_fallback"
            assert repair_payload["summary"]["python_fallback_space_count"] == 1
            assert room["repair_backend"] == "python"
            assert room["repair_status"] == "fallback_python"
            assert room["repair_reason"] == "Exact repair worker unavailable."

            repair_report = json.loads((jobs_root / job_id / "geometry" / "repair_report.json").read_text(encoding="utf-8"))
            assert repair_report["summary"]["python_fallback_space_count"] == 1
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_mocked_exact_repair_worker_preserves_boundary_outputs() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("exact-repair-success")
    fake_worker = make_fake_worker_binary(jobs_root / "fake-worker.exe")

    def fake_invoke_exact_repair_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
        assert worker_binary == fake_worker
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        result_path.write_text(json.dumps(build_exact_repair_response(request_payload), indent=2), encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(geometry_worker, "_invoke_exact_repair_worker", fake_invoke_exact_repair_worker)

    try:
        with build_client(jobs_root, exact_repair_worker_binary=fake_worker) as client:
            created = client.post(
                "/jobs",
                files={"file": ("shared-wall.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = wait_for_output_report(client, job_id)
            assert output_payload["preprocessing"]["worker_backend"] == "hybrid"
            assert output_payload["preprocessing"]["repair"]["summary"]["exact_passthrough_space_count"] == 2
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 1
            assert output_payload["external_candidates"]["summary"]["candidate_surface_count"] == 10
            assert output_payload["external_shell"]["summary"]["candidate_surface_count"] == 10
            assert all(space["repair_backend"] == "cpp-cgal" for space in output_payload["spaces"])
            assert all(space["repair_status"] == "exact_passthrough" for space in output_payload["spaces"])
    finally:
        monkeypatch.undo()
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_malformed_exact_repair_worker_payload_falls_back_to_python() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("exact-repair-malformed")
    fake_worker = make_fake_worker_binary(jobs_root / "fake-worker.exe")

    def fake_invoke_exact_repair_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
        assert worker_binary == fake_worker
        result_path.write_text(
            json.dumps(
                {
                    "contract_version": geometry_worker.EXACT_REPAIR_CONTRACT_VERSION,
                    "status": "ok",
                    "worker_backend": "cpp-cgal",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(geometry_worker, "_invoke_exact_repair_worker", fake_invoke_exact_repair_worker)

    try:
        with build_client(jobs_root, exact_repair_worker_binary=fake_worker) as client:
            created = client.post(
                "/jobs",
                files={"file": ("single-room.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = wait_for_output_report(client, job_id)
            room = output_payload["spaces"][0]
            repair_payload = output_payload["preprocessing"]["repair"]

            assert room["repair_backend"] == "python"
            assert room["repair_status"] == "fallback_python"
            assert "missing the spaces list" in room["repair_reason"]
            assert repair_payload["response_status"] == "failed"
            assert "missing the spaces list" in repair_payload["fallback_reason"]
    finally:
        monkeypatch.undo()
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_openings_remain_not_attempted_when_exact_repair_is_enabled() -> None:
    fixture = build_extraction_fixture()
    jobs_root = make_test_root("exact-repair-opening")
    fake_worker = make_fake_worker_binary(jobs_root / "fake-worker.exe")

    def fake_invoke_exact_repair_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        result_path.write_text(json.dumps(build_exact_repair_response(request_payload), indent=2), encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(geometry_worker, "_invoke_exact_repair_worker", fake_invoke_exact_repair_worker)

    try:
        with build_client(jobs_root, exact_repair_worker_binary=fake_worker) as client:
            created = client.post(
                "/jobs",
                files={"file": ("mixed.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            output_payload = wait_for_output_report(client, job_id)
            opening = next(opening for opening in output_payload["openings"] if opening["name"] == fixture.opening_name)
            assert opening["repair_backend"] == "none"
            assert opening["repair_status"] == "not_attempted"
            assert opening["repair_reason"] == "Exact repair targets IfcSpace only."
    finally:
        monkeypatch.undo()
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_failed_exact_repair_attempt_preserves_failure_metadata() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("exact-repair-failed-space")
    fake_worker = make_fake_worker_binary(jobs_root / "fake-worker.exe")
    self_intersecting_result = normalize_mesh(*build_overlapping_cube_components_mesh())

    def fake_invoke_exact_repair_worker(worker_binary: Path, request_path: Path, result_path: Path) -> None:
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        failed_space = request_payload["spaces"][0]
        result_path.write_text(
            json.dumps(
                {
                    "contract_version": geometry_worker.EXACT_REPAIR_CONTRACT_VERSION,
                    "status": "ok",
                    "worker_backend": "cpp-cgal",
                    "reason": None,
                    "spaces": [
                        {
                            "global_id": failed_space.get("global_id"),
                            "express_id": failed_space["express_id"],
                            "name": failed_space.get("name"),
                            "mesh": None,
                            "vertex_count": 0,
                            "face_count": 0,
                            "component_count": 0,
                            "components": [],
                            "repair_actions": ["cgal_regularization_failed"],
                            "repair_backend": "cpp-cgal",
                            "repair_status": "exact_repaired",
                            "repair_reason": "CGAL repair failed to regularize mesh.",
                            "closed": False,
                            "manifold": False,
                            "outward_normals": False,
                            "volume_m3": 0.0,
                            "valid": False,
                            "reason": "CGAL repair failed to regularize mesh.",
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def synthetic_normalize_mesh(vertices, faces):
        return self_intersecting_result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(geometry_worker, "_invoke_exact_repair_worker", fake_invoke_exact_repair_worker)
    monkeypatch.setattr(geometry_worker, "normalize_mesh", synthetic_normalize_mesh)

    try:
        with build_client(jobs_root, exact_repair_worker_binary=fake_worker) as client:
            created = client.post(
                "/jobs",
                files={"file": ("self-intersection.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            output_payload = wait_for_output_report(client, job_id)
            room = output_payload["spaces"][0]
            assert output_payload["preflight"]["status"] == "failed"
            assert room["repair_backend"] == "python"
            assert room["repair_status"] == "fallback_python"
            assert room["repair_reason"] == "CGAL repair failed to regularize mesh."
            assert room["preflight_failed"] is True
    finally:
        monkeypatch.undo()
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_two_rooms_survive_preprocessing_independently() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("shared-wall-preprocessing")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("two-room.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["summary"]["number_of_spaces"] == 2
            assert output_payload["preprocessing"]["summary"]["valid_spaces"] == 2
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 1
            assert output_payload["external_candidates"]["summary"]["candidate_surface_count"] == 10
            assert output_payload["external_shell"]["summary"]["candidate_surface_count"] == 10

            per_space_artifacts = [space["artifacts"]["obj"] for space in output_payload["spaces"]]
            assert len(per_space_artifacts) == 2
            assert len(set(per_space_artifacts)) == 2
            for artifact_path in per_space_artifacts:
                artifact_response = client.get(f"/jobs/{job_id}/artifacts/{artifact_path}")
                assert artifact_response.status_code == 200
                assert "o " in artifact_response.text

            spaces_all_response = client.get(f"/jobs/{job_id}/artifacts/geometry/spaces_all.obj")
            assert spaces_all_response.status_code == 200
            assert spaces_all_response.text.count("\no ") >= 2

            raw_spaces_response = client.get(f"/jobs/{job_id}/artifacts/geometry/raw/spaces_all.obj")
            assert raw_spaces_response.status_code == 200
            assert raw_spaces_response.text.count("\no ") >= 2
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_shared_wall_model_produces_one_internal_boundary_pair() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("shared-wall-boundary")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("shared-wall.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["preflight"]["status"] == "passed"
            boundaries = output_payload["internal_boundaries"]
            assert boundaries["threshold_m"] == pytest.approx(0.30)
            assert boundaries["summary"]["adjacent_pair_count"] == 1
            assert boundaries["summary"]["shared_surface_count"] == 1

            spaces_by_global_id = {space["global_id"]: space["name"] for space in output_payload["spaces"]}
            adjacency = boundaries["adjacencies"][0]
            adjacency_names = tuple(sorted((spaces_by_global_id[adjacency["space_a_global_id"]], spaces_by_global_id[adjacency["space_b_global_id"]])))
            assert adjacency_names == fixture.expected_adjacency_pairs[0]
            assert adjacency["shared_area_m2"] == pytest.approx(fixture.expected_shared_area_m2, abs=0.05)

            detail_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/internal_boundaries.json").text)
            assert detail_payload["epsilon"] == pytest.approx(1e-3)
            assert detail_payload["summary"]["candidate_pair_count"] == 1
            assert detail_payload["summary"]["adjacent_pair_count"] == 1
            assert detail_payload["summary"]["oriented_surface_count"] == 2
            assert detail_payload["shared_surfaces"][0]["area_m2"] == pytest.approx(fixture.expected_shared_area_m2, abs=0.05)
            assert len(detail_payload["adjacencies"][0]["oriented_surface_ids"]) == 2
            oriented_surfaces = detail_payload["oriented_surfaces"]
            assert len(oriented_surfaces) == 2
            normal_a = oriented_surfaces[0]["plane_normal"]
            normal_b = oriented_surfaces[1]["plane_normal"]
            assert sum(left * right for left, right in zip(normal_a, normal_b, strict=False)) < -0.99

            candidates_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/external_candidates/result.json").text)
            assert candidates_payload["summary"]["candidate_surface_count"] == 10
            assert candidates_payload["summary"]["subtracted_source_polygon_count"] == 2
            assert len(candidates_payload["spaces"]) == 2
            assert all(len(space["candidate_surface_ids"]) == 5 for space in candidates_payload["spaces"])

            obj_response = client.get(f"/jobs/{job_id}/artifacts/geometry/internal_boundaries.obj")
            assert obj_response.status_code == 200
            assert "o ib_0_0" in obj_response.text
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_heuristic_one_room_classifies_outer_surfaces() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("heuristic-one-room")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "heuristic"},
                files={"file": ("one-room.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            shell = output_payload["external_shell"]
            assert shell["mode_requested"] == "heuristic"
            assert shell["mode_effective"] == "heuristic"
            assert shell["summary"]["candidate_surface_count"] == 6
            assert shell["summary"]["internal_partition_match_count"] == 0
            assert shell["summary"]["per_class_counts"]["external_wall"] == 4
            assert shell["summary"]["per_class_counts"]["roof"] == 1
            assert shell["summary"]["per_class_counts"]["ground_floor"] == 1
            assert shell["summary"]["per_class_counts"]["unclassified"] == 0
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_heuristic_mode_does_not_require_native_shell_worker() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("heuristic-without-native-worker")

    try:
        with build_client(jobs_root, shell_worker_binary=jobs_root / "missing-shell-worker.exe") as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "heuristic"},
                files={"file": ("heuristic-no-native.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = wait_for_output_report(client, job_id)
            assert output_payload["success"] is True
            assert output_payload["external_shell"]["mode_requested"] == "heuristic"
            assert output_payload["external_shell"]["mode_effective"] == "heuristic"
            assert output_payload["external_shell"]["shell_backend"] == "python"
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_heuristic_two_room_model_keeps_shared_wall_internal() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("heuristic-two-room-shell")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "heuristic"},
                files={"file": ("two-room.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            shell = output_payload["external_shell"]
            assert output_payload["external_candidates"]["summary"]["candidate_surface_count"] == 10
            assert shell["summary"]["candidate_surface_count"] == 10
            assert shell["summary"]["per_class_counts"]["internal_void"] == 0
            assert shell["summary"]["per_class_counts"]["roof"] == 2
            assert shell["summary"]["per_class_counts"]["ground_floor"] == 2
            assert shell["summary"]["per_class_counts"]["external_wall"] == 6
            assert shell["summary"]["per_class_counts"]["unclassified"] == 0
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_alpha_wrap_request_can_succeed_with_mocked_shell(monkeypatch) -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-success")

    def fake_alpha_wrap(request_payload, worker_binary, workspace_dir):
        shell_payload = external_shell.generate_heuristic_shell(request_payload)
        shell_payload["backend"] = external_shell.ALPHA_WRAP_NATIVE_BACKEND
        return shell_payload

    monkeypatch.setattr(external_shell, "generate_alpha_wrap_shell", fake_alpha_wrap)

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "alpha_wrap"},
                files={"file": ("alpha-success.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            result_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/external_shell/result.json").text)
            assert result_payload["mode_requested"] == "alpha_wrap"
            assert result_payload["mode_effective"] == "alpha_wrap"
            assert result_payload["fallback_reason"] is None
            assert result_payload["shell_backend"] == external_shell.ALPHA_WRAP_NATIVE_BACKEND
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_alpha_wrap_request_fails_when_worker_is_unavailable() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-fallback")

    try:
        with build_client(jobs_root, shell_worker_binary=jobs_root / "missing-shell-worker.exe") as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "alpha_wrap"},
                files={"file": ("alpha-fallback.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            result_payload = wait_for_output_report(client, job_id)
            assert result_payload["success"] is False
            assert result_payload["external_shell"]["mode_requested"] == "alpha_wrap"
            assert result_payload["external_shell"]["mode_effective"] is None
            assert result_payload["external_shell"]["shell_backend"] is None
            assert result_payload["external_shell"]["summary"] == {}
            assert result_payload["external_shell"]["alpha_wrap"]["status"] == "failed"
            assert "unavailable" in (result_payload["error"] or "").lower()
            assert "unavailable" in (result_payload["external_shell"]["alpha_wrap"]["error"] or "").lower()
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_alpha_wrap_request_fails_when_worker_exits_nonzero() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-worker-error")
    failing_worker = write_shell_worker_script(
        jobs_root / "failing-shell-worker.py",
        """
import sys

sys.stderr.write("native alpha-wrap worker crashed")
sys.exit(1)
""",
    )

    try:
        with build_client(jobs_root, shell_worker_binary=failing_worker) as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "alpha_wrap"},
                files={"file": ("alpha-worker-error.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            result_payload = wait_for_output_report(client, job_id)
            assert result_payload["success"] is False
            assert "crashed" in (result_payload["error"] or "").lower()
            assert "crashed" in (result_payload["external_shell"]["alpha_wrap"]["error"] or "").lower()
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_alpha_wrap_request_fails_when_worker_returns_unsupported_contract() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-invalid-contract")
    invalid_worker = write_shell_worker_script(
        jobs_root / "invalid-shell-worker.py",
        """
import json
import sys

request_path = sys.argv[1]
result_path = sys.argv[2]
request_payload = json.loads(open(request_path, encoding="utf-8").read())
payload = {
    "contract_version": 999,
    "status": "ok",
    "backend": "cpp-cgal-alpha-wrap",
    "alpha_m_effective": request_payload.get("alpha_m_effective"),
    "offset_m_effective": request_payload.get("offset_m_effective"),
    "shell_mesh": {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "faces": [[0, 1, 2]],
    },
}
open(result_path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
""",
    )

    try:
        with build_client(jobs_root, shell_worker_binary=invalid_worker) as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "alpha_wrap"},
                files={"file": ("alpha-invalid-contract.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            result_payload = wait_for_output_report(client, job_id)
            assert result_payload["success"] is False
            assert "unsupported contract version" in (result_payload["error"] or "").lower()
            assert "unsupported contract version" in (
                result_payload["external_shell"]["alpha_wrap"]["error"] or ""
            ).lower()
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_partial_alpha_wrap_shell_excludes_missing_regions_as_internal_void(monkeypatch) -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-partial")

    def partial_shell(request_payload, worker_binary, workspace_dir):
        shell_payload = external_shell.generate_heuristic_shell(request_payload)
        shell_mesh = shell_payload["shell_mesh"]
        shell_payload["shell_mesh"] = {
            "vertices": shell_mesh["vertices"],
            "faces": shell_mesh["faces"][:4],
        }
        shell_payload["backend"] = external_shell.ALPHA_WRAP_NATIVE_BACKEND
        return shell_payload

    monkeypatch.setattr(external_shell, "generate_alpha_wrap_shell", partial_shell)

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={"external_shell_mode": "alpha_wrap"},
                files={"file": ("partial-shell.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            result_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/external_shell/result.json").text)
            assert result_payload["mode_effective"] == "alpha_wrap"
            assert result_payload["summary"]["internal_void_count"] > 0
            assert result_payload["summary"]["unclassified_count"] == 0

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["summary"]["unclassified_surface_count"] == 0
            assert any(surface["classification"] == "internal_void" for surface in viewer_manifest["surface_entities"])
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_alpha_wrap_parameters_are_persisted_and_clamped() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-params")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={
                    "alpha_wrap_alpha_m": "0.20",
                    "alpha_wrap_offset_m": "0.02",
                    "internal_boundary_thickness_threshold_m": "0.30",
                },
                files={"file": ("alpha-params.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = wait_for_output_report(client, job_id)
            alpha_wrap = output_payload["external_shell"]["alpha_wrap"]
            assert alpha_wrap["alpha_m_requested"] == pytest.approx(0.20)
            assert alpha_wrap["alpha_m_effective"] == pytest.approx(0.35)
            assert alpha_wrap["offset_m_requested"] == pytest.approx(0.02)
            assert alpha_wrap["offset_m_effective"] == pytest.approx(0.02)

            request_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/external_shell/request.json").text)
            assert request_payload["alpha_m_requested"] == pytest.approx(0.20)
            assert request_payload["alpha_m_effective"] == pytest.approx(0.35)
            assert request_payload["offset_m_requested"] == pytest.approx(0.02)
            assert request_payload["offset_m_effective"] == pytest.approx(0.02)
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_alpha_wrap_offset_must_be_positive() -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("alpha-wrap-invalid-offset")

    try:
        with build_client(jobs_root) as client:
            response = client.post(
                "/jobs",
                data={"alpha_wrap_offset_m": "0"},
                files={"file": ("invalid-offset.ifc", fixture.content, "application/octet-stream")},
            )
            assert response.status_code == 422
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_remove_spaces_rerun_inherits_alpha_wrap_parameters() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("alpha-wrap-rerun")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={
                    "alpha_wrap_alpha_m": "0.20",
                    "alpha_wrap_offset_m": "0.02",
                    "internal_boundary_thickness_threshold_m": "0.30",
                },
                files={"file": ("rerun-alpha.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = wait_for_output_report(client, job_id)
            room = output_payload["spaces"][0]
            rerun_response = client.post(
                f"/jobs/{job_id}/rerun/remove-spaces",
                json={"space_global_ids": [room["global_id"]]},
            )
            assert rerun_response.status_code == 202
            rerun_job_id = rerun_response.json()["job_id"]

            rerun_status, _ = wait_for_terminal_state(client, rerun_job_id)
            assert rerun_status["state"] == "complete"

            rerun_output = wait_for_output_report(client, rerun_job_id)
            alpha_wrap = rerun_output["external_shell"]["alpha_wrap"]
            assert alpha_wrap["alpha_m_requested"] == pytest.approx(0.20)
            assert alpha_wrap["alpha_m_effective"] == pytest.approx(0.35)
            assert alpha_wrap["offset_m_requested"] == pytest.approx(0.02)
            assert alpha_wrap["offset_m_effective"] == pytest.approx(0.02)
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_separated_rooms_produce_zero_internal_boundaries() -> None:
    fixture = build_separated_rooms_fixture()
    jobs_root = make_test_root("separated-rooms")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("separated.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            boundaries = output_payload["internal_boundaries"]
            assert boundaries["summary"]["adjacent_pair_count"] == 0
            assert boundaries["summary"]["shared_surface_count"] == 0
            assert boundaries["adjacencies"] == []
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_threshold_override_detects_internal_boundary_across_wall_gap() -> None:
    fixture = build_separated_rooms_fixture()
    jobs_root = make_test_root("separated-rooms-threshold")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                data={"internal_boundary_thickness_threshold_m": "1.5"},
                files={"file": ("separated-threshold.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            boundaries = output_payload["internal_boundaries"]
            assert boundaries["threshold_m"] == pytest.approx(1.5)
            assert boundaries["summary"]["adjacent_pair_count"] == 1
            assert boundaries["summary"]["shared_surface_count"] == 1

            detail_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/internal_boundaries.json").text)
            assert detail_payload["summary"]["oriented_surface_count"] == 2
            assert len(detail_payload["adjacencies"][0]["oriented_surface_ids"]) == 2

            candidates_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/external_candidates/result.json").text)
            assert candidates_payload["summary"]["candidate_surface_count"] == 10
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_overlapping_rooms_fail_preflight_with_space_clash() -> None:
    fixture = build_overlapping_rooms_fixture()
    jobs_root = make_test_root("overlapping-rooms")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("overlap.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"
            assert "Preflight failed:" in status_payload["error"]

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["success"] is False
            assert output_payload["preflight"]["status"] == "failed"
            assert output_payload["preflight"]["summary"]["clash_pair_count"] == 1
            assert output_payload["preflight"]["summary"]["clash_group_count"] == 1
            clash_blockers = [
                blocker for blocker in output_payload["preflight"]["blockers"] if blocker["code"] == "space_clash"
            ]
            assert len(clash_blockers) == 1
            blocker_names = sorted(ref["name"] for ref in clash_blockers[0]["entities"])
            assert blocker_names == sorted(fixture.space_names)
            assert clash_blockers[0]["classification"] == "partial_overlap"
            assert clash_blockers[0]["recommended_resolution"] is None
            assert output_payload["preflight"]["review_required"] is True
            assert output_payload["internal_boundaries"]["summary"] == {}
            assert output_payload["external_candidates"]["summary"] == {}
            assert output_payload["external_shell"]["summary"] == {}

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["summary"]["failed_count"] == 2
            assert viewer_manifest["summary"]["clash_group_count"] == 1
            assert viewer_manifest["layers"]["normalized_spaces"]["available"] is True
            assert viewer_manifest["layers"]["failed_entities"]["available"] is True

            artifact_names = {item["name"] for item in client.get(f"/jobs/{job_id}/artifacts").json()["artifacts"]}
            assert "geometry/preflight.json" in artifact_names
            assert "geometry/clash_report.json" in artifact_names
            assert "geometry/viewer_manifest.json" in artifact_names
            assert "geometry/spaces_all.obj" in artifact_names
            assert "geometry/internal_boundaries.json" not in artifact_names
            assert "geometry/external_candidates/result.json" not in artifact_names
            assert "geometry/external_shell/result.json" not in artifact_names
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_duplicate_rooms_produce_exact_duplicate_clash_group_with_recommendation() -> None:
    fixture = build_duplicate_rooms_fixture()
    jobs_root = make_test_root("duplicate-rooms")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("duplicate.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            output_payload = wait_for_output_report(client, job_id)
            clash_group = output_payload["preflight"]["clash_groups"][0]
            assert clash_group["classification"] == "exact_duplicate"
            assert clash_group["resolution_status"] == "recommended"
            assert clash_group["recommended_resolution"]["operation"] == "remove_spaces"
            assert len(clash_group["recommended_resolution"]["spaces_to_remove"]) == 1
            assert output_payload["preflight"]["recommended_resolution"]["operation"] == "resolve_space_clashes"

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            manifest_entities = {entity["name"]: entity for entity in viewer_manifest["entities"]}
            assert any(entity["recommended_clash_action"] == "keep" for entity in manifest_entities.values())
            assert any(entity["recommended_clash_action"] == "remove" for entity in manifest_entities.values())
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_contained_fragment_rooms_produce_containment_recommendation() -> None:
    fixture = build_contained_fragment_fixture()
    jobs_root = make_test_root("contained-fragment")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("contained.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            output_payload = wait_for_output_report(client, job_id)
            clash_group = output_payload["preflight"]["clash_groups"][0]
            assert clash_group["classification"] == "contained_fragment"
            assert clash_group["resolution_status"] == "recommended"
            assert clash_group["recommended_resolution"]["keeper"]["name"] == "Outer Space"
            assert clash_group["recommended_resolution"]["spaces_to_remove"][0]["name"] == "Inner Fragment"
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_resolve_space_clashes_rerun_creates_child_job_and_completes() -> None:
    fixture = build_duplicate_rooms_fixture()
    jobs_root = make_test_root("resolve-space-clashes")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("duplicate.ifc", fixture.content, "application/octet-stream")},
            ).json()
            failed_job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, failed_job_id)
            assert status_payload["state"] == "failed"

            failed_output = wait_for_output_report(client, failed_job_id)
            clash_group = failed_output["preflight"]["clash_groups"][0]
            rerun_response = client.post(
                f"/jobs/{failed_job_id}/rerun/resolve-space-clashes",
                json={
                    "group_resolutions": [
                        {
                            "clash_group_id": clash_group["clash_group_id"],
                            "remove_space_global_ids": [
                                space_ref["global_id"]
                                for space_ref in clash_group["recommended_resolution"]["spaces_to_remove"]
                                if space_ref.get("global_id")
                            ],
                            "remove_space_express_ids": [
                                space_ref["express_id"]
                                for space_ref in clash_group["recommended_resolution"]["spaces_to_remove"]
                                if not space_ref.get("global_id")
                            ],
                        }
                    ]
                },
            )
            assert rerun_response.status_code == 202
            rerun_payload = rerun_response.json()
            rerun_job_id = rerun_payload["job_id"]
            assert rerun_payload["parent_job_id"] == failed_job_id
            assert rerun_payload["resolved_clash_group_count"] == 1

            rerun_status, _ = wait_for_terminal_state(client, rerun_job_id)
            assert rerun_status["state"] == "complete"

            rerun_output = wait_for_output_report(client, rerun_job_id)
            assert rerun_output["preflight"]["status"] == "passed"
            assert rerun_output["summary"]["number_of_spaces"] == 1
            assert rerun_output["derivation"]["operation"] == "resolve_space_clashes"
            assert rerun_output["derivation"]["resolved_clash_group_count"] == 1
            assert len(rerun_output["derivation"]["resolved_clash_group_ids"]) == 1

            edit_payload = json.loads((jobs_root / rerun_job_id / "edits" / "resolve_space_clashes.json").read_text(encoding="utf-8"))
            assert edit_payload["parent_job_id"] == failed_job_id
            assert edit_payload["operation"] == "resolve_space_clashes"
            assert edit_payload["resolved_clash_group_count"] == 1
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_corridor_only_reports_touching_space_pairs() -> None:
    fixture = build_corridor_room_fixture()
    jobs_root = make_test_root("corridor-room")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("corridor.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            boundaries = output_payload["internal_boundaries"]
            assert boundaries["summary"]["adjacent_pair_count"] == 1
            assert boundaries["summary"]["shared_surface_count"] == 1

            spaces_by_global_id = {space["global_id"]: space["name"] for space in output_payload["spaces"]}
            adjacency = boundaries["adjacencies"][0]
            adjacency_names = tuple(sorted((spaces_by_global_id[adjacency["space_a_global_id"]], spaces_by_global_id[adjacency["space_b_global_id"]])))
            assert adjacency_names == fixture.expected_adjacency_pairs[0]
            assert adjacency["shared_area_m2"] == pytest.approx(fixture.expected_shared_area_m2, abs=0.05)
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_missing_space_representation_fails_preflight_and_preserves_viewer_artifacts() -> None:
    fixture = build_extraction_fixture()
    jobs_root = make_test_root("mixed")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("mixed.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"
            assert "Preflight failed:" in status_payload["error"]

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["success"] is False
            assert output_payload["summary"]["number_of_spaces"] == fixture.expected_space_count
            assert output_payload["summary"]["number_of_openings"] == fixture.expected_opening_count
            assert output_payload["preprocessing"]["worker_backend"] == "python"
            assert output_payload["preflight"]["status"] == "failed"
            assert output_payload["preflight"]["summary"]["blocker_count"] == 1
            assert output_payload["preflight"]["blockers"][0]["code"] == "invalid_space_solid"
            assert output_payload["preflight"]["blockers"][0]["entity"]["name"] == fixture.missing_space_name
            assert output_payload["internal_boundaries"]["summary"] == {}
            assert output_payload["external_candidates"]["summary"] == {}
            assert output_payload["external_shell"]["summary"] == {}

            spaces_by_name = {space["name"]: space for space in output_payload["spaces"]}
            openings_by_name = {opening["name"]: opening for opening in output_payload["openings"]}

            represented_space = spaces_by_name[fixture.represented_space_name]
            assert represented_space["geometry_ok"] is True
            assert represented_space["closed"] is True

            missing_space = spaces_by_name[fixture.missing_space_name]
            assert missing_space["geometry_ok"] is False
            assert missing_space["geometry_error"] == "Missing representation"
            assert missing_space["preflight_failed"] is True
            assert missing_space["preflight_reason"] == "Normalized space solid is missing or topologically invalid."

            opening = openings_by_name[fixture.opening_name]
            assert opening["geometry_ok"] is True
            assert opening["storey"]["name"] == fixture.storey_name
            assert opening["artifacts"]["normalized_obj"]
            assert opening["artifacts"]["raw_obj"]

            geometry_summary = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/geometry_summary.json").text)
            invalid_names = {entity["name"]: entity["reason"] for entity in geometry_summary["invalid_entities"]}
            assert invalid_names[fixture.missing_space_name] == "Missing representation"

            preflight_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/preflight.json").text)
            assert preflight_payload["status"] == "failed"
            skipped_names = {
                entry["entity"]["name"]: entry["reason"]
                for entry in preflight_payload["warnings"]
                if entry["code"] == "space_skipped_nonblocking"
            }
            assert "Missing representation" in skipped_names[fixture.missing_space_name]

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["summary"]["failed_count"] == 1
            assert "surface_entities" in viewer_manifest
            missing_manifest_entity = next(
                entity for entity in viewer_manifest["entities"] if entity["name"] == fixture.missing_space_name
            )
            assert missing_manifest_entity["failed"] is True
            assert missing_manifest_entity["artifacts"]["raw_obj"] is None
            assert missing_manifest_entity["artifacts"]["normalized_obj"] is None
            assert missing_manifest_entity["marker_origin"] is not None

            represented_manifest_entity = next(
                entity for entity in viewer_manifest["entities"] if entity["name"] == fixture.represented_space_name
            )
            assert represented_manifest_entity["artifacts"]["raw_obj"]
            assert represented_manifest_entity["artifacts"]["normalized_obj"]
            assert represented_manifest_entity["object_name"] == represented_space["global_id"]

            artifacts_response = client.get(f"/jobs/{job_id}/artifacts")
            artifact_names = {item["name"] for item in artifacts_response.json()["artifacts"]}
            assert "geometry/viewer_manifest.json" in artifact_names
            assert "geometry/preflight.json" in artifact_names
            assert "geometry/raw/spaces_all.obj" in artifact_names
            assert "geometry/raw/openings.obj" in artifact_names
            assert "geometry/openings.obj" in artifact_names
            assert "geometry/spaces_all.obj" in artifact_names
            assert "geometry/internal_boundaries.json" not in artifact_names
            assert "geometry/external_candidates/result.json" not in artifact_names
            assert "geometry/external_shell/result.json" not in artifact_names
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_per_entity_tessellation_failure_fails_preflight_and_keeps_viewer_manifest(monkeypatch) -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("tessellation-failure")

    original_build_shape = geometry_worker.build_shape

    def flaky_shape_builder(settings, entity):
        if entity.is_a("IfcSpace") and entity.Name == fixture.represented_space_name:
            raise RuntimeError("synthetic tessellation failure")
        return original_build_shape(settings, entity)

    monkeypatch.setattr(geometry_worker, "build_shape", flaky_shape_builder)

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("two-room.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"
            assert "Preflight failed:" in status_payload["error"]

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["success"] is False
            assert output_payload["preprocessing"]["summary"]["valid_entities"] == 1
            assert output_payload["preprocessing"]["summary"]["invalid_entities"] == 1
            assert output_payload["preflight"]["status"] == "failed"
            assert output_payload["preflight"]["summary"]["blocker_count"] == 1
            assert output_payload["internal_boundaries"]["summary"] == {}
            assert output_payload["external_candidates"]["summary"] == {}
            assert output_payload["external_shell"]["summary"] == {}

            spaces_by_name = {space["name"]: space for space in output_payload["spaces"]}
            failed_space = spaces_by_name[fixture.represented_space_name]
            assert failed_space["geometry_ok"] is False
            assert "synthetic tessellation failure" in failed_space["geometry_error"]
            assert failed_space["artifacts"]["raw_obj"] is None
            assert failed_space["artifacts"]["normalized_obj"] is None
            assert failed_space["preflight_failed"] is True

            surviving_spaces = [space for space in output_payload["spaces"] if space["geometry_ok"]]
            assert len(surviving_spaces) == 1
            assert "obj" in surviving_spaces[0]["artifacts"]

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            failed_manifest_entity = next(
                entity for entity in viewer_manifest["entities"] if entity["name"] == fixture.represented_space_name
            )
            assert failed_manifest_entity["failed"] is True
            assert failed_manifest_entity["marker_origin"] is not None
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_synthetic_self_intersecting_space_fails_preflight(monkeypatch) -> None:
    fixture = build_single_room_fixture()
    jobs_root = make_test_root("self-intersection")
    self_intersecting_result = normalize_mesh(*build_overlapping_cube_components_mesh())

    def synthetic_normalize_mesh(vertices, faces):
        return self_intersecting_result

    monkeypatch.setattr(geometry_worker, "normalize_mesh", synthetic_normalize_mesh)

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("self-intersection.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"
            assert "Preflight failed:" in status_payload["error"]

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["preflight"]["status"] == "failed"
            blockers = output_payload["preflight"]["blockers"]
            assert [blocker["code"] for blocker in blockers] == ["self_intersection"]

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["summary"]["failed_count"] == 1
            manifest_space = viewer_manifest["entities"][0]
            assert manifest_space["failed"] is True
            assert manifest_space["reason"] == "Normalized space mesh contains intersecting triangles."
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_remove_spaces_rerun_creates_child_job_with_lineage_and_filtered_input() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("remove-spaces-child")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("shared-wall.ifc", fixture.content, "application/octet-stream")},
            ).json()
            parent_job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, parent_job_id)
            assert status_payload["state"] == "complete"

            parent_output = wait_for_output_report(client, parent_job_id)
            removed_space = next(space for space in parent_output["spaces"] if space["name"] == fixture.represented_space_name)

            rerun_response = client.post(
                f"/jobs/{parent_job_id}/rerun/remove-spaces",
                json={"space_global_ids": [removed_space["global_id"]]},
            )
            assert rerun_response.status_code == 202
            rerun_payload = rerun_response.json()
            child_job_id = rerun_payload["job_id"]
            assert rerun_payload["parent_job_id"] == parent_job_id
            assert rerun_payload["root_job_id"] == parent_job_id
            assert rerun_payload["removed_space_count"] == 1

            child_status, _ = wait_for_terminal_state(client, child_job_id)
            assert child_status["state"] == "complete"
            assert child_status["derivation"]["parent_job_id"] == parent_job_id

            child_output = wait_for_output_report(client, child_job_id)
            assert child_output["summary"]["number_of_spaces"] == 1
            assert child_output["derivation"]["parent_job_id"] == parent_job_id
            assert child_output["derivation"]["root_job_id"] == parent_job_id
            assert child_output["derivation"]["operation"] == "remove_spaces"
            assert child_output["derivation"]["removed_spaces"][0]["global_id"] == removed_space["global_id"]

            artifacts = {artifact["name"] for artifact in client.get(f"/jobs/{child_job_id}/artifacts").json()["artifacts"]}
            assert "input.ifc" in artifacts
            assert "edits/remove_spaces.json" in artifacts

            derived_model = ifcopenshell.open(str(jobs_root / child_job_id / "input.ifc"))
            assert len(derived_model.by_type("IfcSpace")) == 1

            edit_payload = json.loads((jobs_root / child_job_id / "edits" / "remove_spaces.json").read_text(encoding="utf-8"))
            assert edit_payload["parent_job_id"] == parent_job_id
            assert edit_payload["requested_space_global_ids"] == [removed_space["global_id"]]
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_remove_spaces_rerun_uses_current_job_input_for_cumulative_edits() -> None:
    fixture = build_corridor_room_fixture()
    jobs_root = make_test_root("remove-spaces-cumulative")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("corridor.ifc", fixture.content, "application/octet-stream")},
            ).json()
            root_job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, root_job_id)
            assert status_payload["state"] == "complete"
            root_output = wait_for_output_report(client, root_job_id)

            first_removed = next(space for space in root_output["spaces"] if space["name"] == "Corridor")
            first_rerun = client.post(
                f"/jobs/{root_job_id}/rerun/remove-spaces",
                json={"space_global_ids": [first_removed["global_id"]]},
            )
            assert first_rerun.status_code == 202
            first_child_id = first_rerun.json()["job_id"]
            first_child_status, _ = wait_for_terminal_state(client, first_child_id)
            assert first_child_status["state"] == "complete"

            first_child_output = wait_for_output_report(client, first_child_id)
            assert first_child_output["summary"]["number_of_spaces"] == 2
            assert {space["name"] for space in first_child_output["spaces"]} == {"Touching Room", "Far Room"}

            second_removed = next(space for space in first_child_output["spaces"] if space["name"] == "Touching Room")
            second_rerun = client.post(
                f"/jobs/{first_child_id}/rerun/remove-spaces",
                json={"space_global_ids": [second_removed["global_id"]]},
            )
            assert second_rerun.status_code == 202
            second_child_payload = second_rerun.json()
            second_child_id = second_child_payload["job_id"]
            assert second_child_payload["parent_job_id"] == first_child_id
            assert second_child_payload["root_job_id"] == root_job_id

            second_child_status, _ = wait_for_terminal_state(client, second_child_id)
            assert second_child_status["state"] == "complete"
            second_child_output = wait_for_output_report(client, second_child_id)
            assert second_child_output["summary"]["number_of_spaces"] == 1
            assert second_child_output["derivation"]["parent_job_id"] == first_child_id
            assert second_child_output["derivation"]["root_job_id"] == root_job_id
            assert [space["name"] for space in second_child_output["spaces"]] == ["Far Room"]
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_remove_spaces_rerun_rejects_invalid_requests() -> None:
    fixture = build_extraction_fixture()
    jobs_root = make_test_root("remove-spaces-invalid")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("mixed.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"

            output_payload = wait_for_output_report(client, job_id)
            opening = next(opening for opening in output_payload["openings"] if opening["name"] == fixture.opening_name)

            empty_response = client.post(f"/jobs/{job_id}/rerun/remove-spaces", json={})
            assert empty_response.status_code == 422
            assert "Select at least one IfcSpace to remove." in empty_response.json()["detail"]

            unknown_response = client.post(
                f"/jobs/{job_id}/rerun/remove-spaces",
                json={"space_global_ids": ["does-not-exist"]},
            )
            assert unknown_response.status_code == 422
            assert "Unknown space_global_ids" in unknown_response.json()["detail"]

            non_space_response = client.post(
                f"/jobs/{job_id}/rerun/remove-spaces",
                json={"space_express_ids": [opening["express_id"]]},
            )
            assert non_space_response.status_code == 422
            assert "Non-IfcSpace space_express_ids" in non_space_response.json()["detail"]
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_remove_spaces_rerun_rejects_non_terminal_and_remove_all_requests() -> None:
    fixture = build_shared_wall_fixture()
    jobs_root = make_test_root("remove-spaces-validation")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("shared-wall.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            non_terminal_response = client.post(
                f"/jobs/{job_id}/rerun/remove-spaces",
                json={"space_global_ids": ["not-needed-yet"]},
            )
            assert non_terminal_response.status_code == 409

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"
            output_payload = wait_for_output_report(client, job_id)
            all_space_ids = [space["global_id"] for space in output_payload["spaces"]]

            remove_all_response = client.post(
                f"/jobs/{job_id}/rerun/remove-spaces",
                json={"space_global_ids": all_space_ids},
            )
            assert remove_all_response.status_code == 422
            assert "Cannot remove all remaining IfcSpace entities." in remove_all_response.json()["detail"]
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_remove_spaces_rerun_can_resolve_preflight_clash_failure() -> None:
    fixture = build_overlapping_rooms_fixture()
    jobs_root = make_test_root("remove-spaces-preflight")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("overlap.ifc", fixture.content, "application/octet-stream")},
            ).json()
            failed_job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, failed_job_id)
            assert status_payload["state"] == "failed"

            failed_output = wait_for_output_report(client, failed_job_id)
            assert failed_output["preflight"]["summary"]["blocker_count"] == 1
            removed_space = failed_output["spaces"][0]

            rerun_response = client.post(
                f"/jobs/{failed_job_id}/rerun/remove-spaces",
                json={"space_global_ids": [removed_space["global_id"]]},
            )
            assert rerun_response.status_code == 202
            rerun_job_id = rerun_response.json()["job_id"]

            rerun_status, _ = wait_for_terminal_state(client, rerun_job_id)
            assert rerun_status["state"] == "complete"

            rerun_output = wait_for_output_report(client, rerun_job_id)
            assert rerun_output["preflight"]["status"] == "passed"
            assert rerun_output["preflight"]["summary"]["blocker_count"] == 0
            assert rerun_output["summary"]["number_of_spaces"] == 1
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_invalid_ifc_input_fails_job_cleanly() -> None:
    jobs_root = make_test_root("invalid-ifc")

    try:
        with build_client(jobs_root) as client:
            response = client.post(
                "/jobs",
                files={"file": ("invalid.ifc", b"not an ifc file", "application/octet-stream")},
            )
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "failed"
            assert "Unable to parse IFC SPF header" in status_payload["error"]

            output_payload = json.loads((jobs_root / job_id / "output.json").read_text(encoding="utf-8"))
            assert output_payload["success"] is False
            assert output_payload["error"] == status_payload["error"]

            debug_payload = json.loads((jobs_root / job_id / "debug.json").read_text(encoding="utf-8"))
            assert debug_payload["state"] == "failed"
            assert debug_payload["error"] == status_payload["error"]
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_empty_ifc_returns_zero_counts_without_crashing() -> None:
    fixture = build_empty_fixture()
    jobs_root = make_test_root("empty")

    try:
        with build_client(jobs_root) as client:
            created = client.post(
                "/jobs",
                files={"file": ("empty.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            status_payload, _ = wait_for_terminal_state(client, job_id)
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["schema"] == "IFC4"
            assert output_payload["summary"]["number_of_spaces"] == 0
            assert output_payload["summary"]["number_of_openings"] == 0
            assert output_payload["spaces"] == []
            assert output_payload["openings"] == []
            assert output_payload["preprocessing"]["summary"]["entities_total"] == 0
            assert output_payload["preflight"]["status"] == "passed"
            assert output_payload["preflight"]["summary"]["blocker_count"] == 0
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 0
            assert output_payload["external_candidates"]["summary"]["candidate_surface_count"] == 0
            assert output_payload["external_shell"]["summary"]["candidate_surface_count"] == 0

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["entities"] == []
            assert viewer_manifest["surface_entities"] == []
            assert viewer_manifest["layers"]["failed_entities"]["available"] is False
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_unknown_jobs_and_invalid_artifact_paths_return_404() -> None:
    fixture = build_empty_fixture()
    jobs_root = make_test_root("not-found")

    try:
        with build_client(jobs_root) as client:
            missing_job = client.get("/jobs/does-not-exist")
            assert missing_job.status_code == 404

            created = client.post(
                "/jobs",
                files={"file": ("empty.ifc", fixture.content, "application/octet-stream")},
            ).json()
            job_id = created["job_id"]

            invalid_artifact = client.get(f"/jobs/{job_id}/artifacts/not-real.txt")
            assert invalid_artifact.status_code == 404

            traversal_attempt = client.get(f"/jobs/{job_id}/artifacts/../debug.json")
            assert traversal_attempt.status_code == 404
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def build_overlapping_cube_components_mesh() -> tuple[list[list[float]], list[list[int]]]:
    left_vertices, left_faces = build_cube_mesh()
    right_vertices, right_faces = build_cube_mesh(origin=(0.6, 0.0, 0.0))
    return merge_meshes((left_vertices, left_faces), (right_vertices, right_faces))


def build_cube_mesh(
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[list[list[float]], list[list[int]]]:
    origin_x, origin_y, origin_z = origin
    size_x, size_y, size_z = size
    vertices = [
        [origin_x, origin_y, origin_z],
        [origin_x + size_x, origin_y, origin_z],
        [origin_x + size_x, origin_y + size_y, origin_z],
        [origin_x, origin_y + size_y, origin_z],
        [origin_x, origin_y, origin_z + size_z],
        [origin_x + size_x, origin_y, origin_z + size_z],
        [origin_x + size_x, origin_y + size_y, origin_z + size_z],
        [origin_x, origin_y + size_y, origin_z + size_z],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [0, 7, 3],
        [0, 4, 7],
        [3, 6, 2],
        [3, 7, 6],
    ]
    return vertices, faces


def merge_meshes(
    left_mesh: tuple[list[list[float]], list[list[int]]],
    right_mesh: tuple[list[list[float]], list[list[int]]],
) -> tuple[list[list[float]], list[list[int]]]:
    left_vertices, left_faces = left_mesh
    right_vertices, right_faces = right_mesh
    vertex_offset = len(left_vertices)
    return (
        left_vertices + right_vertices,
        left_faces + [[index + vertex_offset for index in face] for face in right_faces],
    )
