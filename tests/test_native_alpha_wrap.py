from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.ifc_factory import build_shared_wall_fixture


def test_native_alpha_wrap_shared_wall_fixture_completes_with_classified_shell(tmp_path) -> None:
    settings = Settings(jobs_root=tmp_path, stage_delay_seconds=0.0)
    if settings.shell_worker_binary is None or not settings.shell_worker_binary.exists():
        pytest.skip("Native alpha-wrap shell worker is not built.")

    fixture = build_shared_wall_fixture()
    app = create_app(settings)

    with TestClient(app) as client:
        created_response = client.post(
            "/jobs",
            data={"external_shell_mode": "alpha_wrap"},
            files={"file": ("native-alpha-shared-wall.ifc", fixture.content, "application/octet-stream")},
        )
        assert created_response.status_code == 202
        job_id = created_response.json()["job_id"]

        deadline = time.time() + 30.0
        while time.time() < deadline:
            status_response = client.get(f"/jobs/{job_id}")
            assert status_response.status_code == 200
            status_payload = status_response.json()
            if status_payload["state"] in {"complete", "failed"}:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Timed out waiting for native alpha-wrap job to finish.")

        assert status_payload["state"] == "complete", status_payload.get("error")

        output_response = client.get(f"/jobs/{job_id}/artifacts/output.json")
        assert output_response.status_code == 200
        output_payload = output_response.json()
        shell = output_payload["external_shell"]
        alpha_wrap = shell["alpha_wrap"]

        assert output_payload["preflight"]["status"] == "passed"
        assert shell["mode_requested"] == "alpha_wrap"
        assert shell["mode_effective"] == "alpha_wrap"
        assert shell["shell_backend"] == "cpp-cgal-alpha-wrap"
        assert alpha_wrap["status"] == "ok"
        assert alpha_wrap["backend"] == "cpp-cgal-alpha-wrap"
        assert alpha_wrap["triangle_count"] > 0
        assert alpha_wrap["vertex_count"] > 0
        assert shell["summary"]["candidate_surface_count"] == 10
        assert shell["summary"]["shell_match_count"] == 10
        assert shell["summary"]["unclassified_count"] == 0
        assert shell["summary"]["per_class_counts"]["external_wall"] == 6
        assert shell["summary"]["per_class_counts"]["roof"] == 2
        assert shell["summary"]["per_class_counts"]["ground_floor"] == 2
        assert output_payload["gbxml_preflight"]["summary"]["blocker_count"] == 0

        result_response = client.get(f"/jobs/{job_id}/artifacts/geometry/external_shell/result.json")
        assert result_response.status_code == 200
        result_payload = json.loads(result_response.text)
        assert result_payload["alpha_wrap"]["status"] == "ok"

        for artifact_path in (
            "geometry/external_shell/shell.obj",
            "geometry/external_shell/surfaces_all.obj",
            "geometry/2lsb_surfaces.xml",
            "geometry/2lsb_surfaces.obj",
            "geometry/2lsb_surfaces.gbxml",
        ):
            artifact_response = client.get(f"/jobs/{job_id}/artifacts/{artifact_path}")
            assert artifact_response.status_code == 200
