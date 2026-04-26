from __future__ import annotations

from tools.energyplus_stress_case import run_energyplus_stress_debug


def test_energyplus_stress_case_flags_expected_preflight_faults(tmp_path) -> None:
    manifest = run_energyplus_stress_debug(
        tmp_path / "energyplus_fault_stress.ifc",
        tmp_path / "energyplus_fault_stress_debug.json",
        jobs_root=tmp_path / "jobs",
    )

    assert (tmp_path / "energyplus_fault_stress.ifc").exists()
    assert (tmp_path / "energyplus_fault_stress_debug.json").exists()
    assert {check["status"] for check in manifest["checks"]} == {"pass"}

    app_run = manifest["app_run"]
    assert app_run["state"] == "failed"
    assert "preflight" in app_run["history_states"]
    assert "internal_boundary" not in app_run["history_states"]

    preflight = app_run["preflight"]
    summary = preflight["summary"]
    assert summary["blocker_count"] == 4
    assert summary["clash_group_count"] == 3
    assert summary["exact_duplicate_group_count"] == 1
    assert summary["partial_overlap_group_count"] == 1
    assert summary["contained_fragment_group_count"] == 1
    assert summary["recommended_resolution_group_count"] == 2
    assert summary["manual_resolution_group_count"] == 1

    blocker_codes = {blocker["code"] for blocker in preflight["blockers"]}
    clash_classes = {group["classification"] for group in preflight["clash_groups"]}
    assert "invalid_space_solid" in blocker_codes
    assert {"exact_duplicate", "partial_overlap", "contained_fragment"}.issubset(clash_classes)
