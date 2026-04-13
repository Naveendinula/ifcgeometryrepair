from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app.geometry_worker as geometry_worker
from app.config import Settings
from app.main import create_app
from tests.ifc_factory import (
    build_corridor_room_fixture,
    build_empty_fixture,
    build_extraction_fixture,
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


def build_client(jobs_root: Path) -> TestClient:
    app = create_app(
        Settings(
            jobs_root=jobs_root,
            stage_delay_seconds=0.02,
            geometry_worker_binary=jobs_root / "missing-worker.exe",
            internal_boundary_thickness_threshold_m=0.30,
        )
    )
    return TestClient(app)


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
                "internal_boundary",
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
                "geometry/internal_boundaries.json",
                "geometry/internal_boundaries.obj",
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
            assert output_payload["preprocessing"]["summary"]["valid_entities"] == 1
            assert output_payload["preprocessing"]["artifacts"]["viewer_manifest"] == "geometry/viewer_manifest.json"
            assert output_payload["preprocessing"]["artifacts"]["raw_spaces_all"] == "geometry/raw/spaces_all.obj"
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 0
            assert output_payload["internal_boundaries"]["artifacts"]["detail"] == "geometry/internal_boundaries.json"

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

            viewer_manifest = json.loads((job_dir / "geometry" / "viewer_manifest.json").read_text(encoding="utf-8"))
            assert viewer_manifest["summary"]["space_count"] == 1
            assert viewer_manifest["layers"]["raw_ifc_preview"]["available"] is True
            assert viewer_manifest["layers"]["normalized_spaces"]["available"] is True
            assert viewer_manifest["layers"]["failed_entities"]["available"] is False
            assert viewer_manifest["entities"][0]["global_id"] == room["global_id"]

            raw_spaces_text = (job_dir / "geometry" / "raw" / "spaces_all.obj").read_text(encoding="utf-8")
            normalized_spaces_text = (job_dir / "geometry" / "spaces_all.obj").read_text(encoding="utf-8")
            assert f"o {room['global_id']}" in raw_spaces_text
            assert f"o {room['global_id']}" in normalized_spaces_text
    finally:
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
            assert detail_payload["summary"]["candidate_pair_count"] == 1
            assert detail_payload["summary"]["adjacent_pair_count"] == 1
            assert detail_payload["shared_surfaces"][0]["area_m2"] == pytest.approx(fixture.expected_shared_area_m2, abs=0.05)

            obj_response = client.get(f"/jobs/{job_id}/artifacts/geometry/internal_boundaries.obj")
            assert obj_response.status_code == 200
            assert "o ib_0_0" in obj_response.text
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


def test_space_opening_and_missing_representation_are_reported_cleanly() -> None:
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
            assert status_payload["state"] == "complete"

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["summary"]["number_of_spaces"] == fixture.expected_space_count
            assert output_payload["summary"]["number_of_openings"] == fixture.expected_opening_count
            assert output_payload["preprocessing"]["worker_backend"] == "python"
            assert output_payload["internal_boundaries"]["summary"]["skipped_space_count"] == 1

            spaces_by_name = {space["name"]: space for space in output_payload["spaces"]}
            openings_by_name = {opening["name"]: opening for opening in output_payload["openings"]}

            represented_space = spaces_by_name[fixture.represented_space_name]
            assert represented_space["geometry_ok"] is True
            assert represented_space["closed"] is True

            missing_space = spaces_by_name[fixture.missing_space_name]
            assert missing_space["geometry_ok"] is False
            assert missing_space["geometry_error"] == "Missing representation"

            opening = openings_by_name[fixture.opening_name]
            assert opening["geometry_ok"] is True
            assert opening["storey"]["name"] == fixture.storey_name
            assert opening["artifacts"]["normalized_obj"]
            assert opening["artifacts"]["raw_obj"]

            geometry_summary = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/geometry_summary.json").text)
            invalid_names = {entity["name"]: entity["reason"] for entity in geometry_summary["invalid_entities"]}
            assert invalid_names[fixture.missing_space_name] == "Missing representation"

            internal_boundaries = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/internal_boundaries.json").text)
            skipped_names = {entity["name"]: entity["reason"] for entity in internal_boundaries["skipped_spaces"]}
            assert skipped_names[fixture.missing_space_name] == "Missing representation"

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["summary"]["failed_count"] == 1
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
            assert "geometry/internal_boundaries.json" in artifact_names
            assert "geometry/internal_boundaries.obj" in artifact_names
            assert "geometry/raw/spaces_all.obj" in artifact_names
            assert "geometry/raw/openings.obj" in artifact_names
            assert "geometry/openings.obj" in artifact_names
            assert "geometry/spaces_all.obj" in artifact_names
    finally:
        shutil.rmtree(jobs_root, ignore_errors=True)


def test_per_entity_tessellation_failure_does_not_fail_job(monkeypatch) -> None:
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
            assert status_payload["state"] == "complete"
            assert status_payload["error"] is None

            output_payload = json.loads(client.get(f"/jobs/{job_id}/artifacts/output.json").text)
            assert output_payload["preprocessing"]["summary"]["valid_entities"] == 1
            assert output_payload["preprocessing"]["summary"]["invalid_entities"] == 1
            assert output_payload["internal_boundaries"]["summary"]["processed_space_count"] == 1
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 0

            spaces_by_name = {space["name"]: space for space in output_payload["spaces"]}
            failed_space = spaces_by_name[fixture.represented_space_name]
            assert failed_space["geometry_ok"] is False
            assert "synthetic tessellation failure" in failed_space["geometry_error"]
            assert failed_space["artifacts"]["raw_obj"] is None
            assert failed_space["artifacts"]["normalized_obj"] is None

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
            assert output_payload["internal_boundaries"]["summary"]["adjacent_pair_count"] == 0

            viewer_manifest = json.loads(client.get(f"/jobs/{job_id}/artifacts/geometry/viewer_manifest.json").text)
            assert viewer_manifest["entities"] == []
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
