from __future__ import annotations

import pytest

from app.viewer_manifest import build_viewer_manifest


def _minimal_output_payload(**overrides):
    base = {
        "schema": "IFC4",
        "summary": {"number_of_spaces": 2, "number_of_openings": 1},
        "spaces": [],
        "openings": [],
        "preprocessing": {"unit": "meter", "artifacts": {}},
        "preflight": {"clash_groups": []},
        "internal_boundaries": {
            "threshold_m": 0.3,
            "summary": {},
            "adjacencies": [],
            "artifacts": {"obj": "geometry/internal_boundaries.obj"},
        },
        "external_candidates": {},
        "external_shell": {
            "surfaces": [
                {
                    "surface_id": "surf_1",
                    "classification": "external_wall",
                    "area_m2": 5.0,
                    "normal": [1, 0, 0],
                    "centroid": [1.0, 2.0, 3.0],
                    "artifacts": {},
                },
                {
                    "surface_id": "surf_2",
                    "classification": "internal_void",
                    "area_m2": 0.1,
                    "normal": [0, 1, 0],
                    "centroid": [4.0, 5.0, 6.0],
                    "artifacts": {},
                },
            ],
            "artifacts": {"surfaces_all": "geometry/surfaces_all.obj"},
        },
        "opening_integration": {
            "summary": {},
            "opening_surfaces": [
                {
                    "surface_id": "opening_1",
                    "opening_name": "Window_1",
                    "boundary_surface_id": "surf_1",
                    "boundary_classification": "external_wall",
                    "area_m2": 1.5,
                    "normal": [1, 0, 0],
                    "centroid": [1.0, 2.0, 3.5],
                },
            ],
            "artifacts": {"obj": "geometry/opening_integration/openings_projected.obj"},
        },
    }
    base.update(overrides)
    return base


def _sample_oriented_surfaces():
    return [
        {
            "oriented_surface_id": "ib_0_left_0",
            "object_name": "ib_0_left_0",
            "space_global_id": "space_A",
            "space_express_id": 100,
            "space_name": "Room A",
            "adjacent_space_global_id": "space_B",
            "adjacent_space_express_id": 200,
            "adjacent_space_name": "Room B",
            "paired_surface_id": "ib_0_right_0",
            "shared_surface_id": "ib_shared_0",
            "area_m2": 8.0,
            "plane_normal": [1.0, 0.0, 0.0],
            "plane_point": [5.0, 0.0, 0.0],
            "centroid": [5.0, 1.0, 1.5],
        },
        {
            "oriented_surface_id": "ib_0_right_0",
            "object_name": "ib_0_right_0",
            "space_global_id": "space_B",
            "space_express_id": 200,
            "space_name": "Room B",
            "adjacent_space_global_id": "space_A",
            "adjacent_space_express_id": 100,
            "adjacent_space_name": "Room A",
            "paired_surface_id": "ib_0_left_0",
            "shared_surface_id": "ib_shared_0",
            "area_m2": 8.0,
            "plane_normal": [-1.0, 0.0, 0.0],
            "plane_point": [5.3, 0.0, 0.0],
            "centroid": [5.3, 1.0, 1.5],
        },
    ]


class TestBuildViewerManifest:
    def test_minimal_output_produces_valid_manifest(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload())
        assert manifest["job_id"] == "job_1"
        assert manifest["schema"] == "IFC4"
        assert manifest["unit"] == "meter"

    def test_surface_entities_have_below_area_threshold(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload(), min_area_threshold_m2=0.25)
        surfaces = manifest["surface_entities"]
        large = next(s for s in surfaces if s["surface_id"] == "surf_1")
        small = next(s for s in surfaces if s["surface_id"] == "surf_2")
        assert large["below_area_threshold"] is False
        assert small["below_area_threshold"] is True

    def test_opening_surface_entities_included(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload())
        opening_surfaces = manifest["opening_surface_entities"]
        assert len(opening_surfaces) == 1
        assert opening_surfaces[0]["surface_id"] == "opening_1"
        assert opening_surfaces[0]["selection_type"] == "opening_surface"
        assert opening_surfaces[0]["entity_type"] == "OpeningSurface"

    def test_internal_boundary_layer_available_when_payload_provided(self):
        manifest = build_viewer_manifest(
            "job_1",
            _minimal_output_payload(),
            internal_boundaries_payload={
                "oriented_surfaces": _sample_oriented_surfaces(),
                "artifacts": {"obj": "geometry/internal_boundaries.obj"},
            },
        )
        layer = manifest["layers"]["internal_boundaries"]
        assert layer["available"] is True
        assert layer["obj"] == "geometry/internal_boundaries.obj"
        assert layer["count"] == 2

    def test_internal_boundary_entities_have_expected_fields(self):
        manifest = build_viewer_manifest(
            "job_1",
            _minimal_output_payload(),
            internal_boundaries_payload={
                "oriented_surfaces": _sample_oriented_surfaces(),
            },
        )
        entities = manifest["internal_boundary_entities"]
        assert len(entities) == 2
        left = next(e for e in entities if e["surface_id"] == "ib_0_left_0")
        assert left["selection_type"] == "internal_boundary"
        assert left["entity_type"] == "InternalBoundary"
        assert left["space_global_id"] == "space_A"
        assert left["adjacent_space_global_id"] == "space_B"
        assert left["paired_surface_id"] == "ib_0_right_0"
        assert left["classification"] == "internal"
        assert left["area_m2"] == 8.0

    def test_thickness_computed_from_paired_surfaces(self):
        manifest = build_viewer_manifest(
            "job_1",
            _minimal_output_payload(),
            internal_boundaries_payload={
                "oriented_surfaces": _sample_oriented_surfaces(),
            },
        )
        entities = manifest["internal_boundary_entities"]
        left = next(e for e in entities if e["surface_id"] == "ib_0_left_0")
        assert left["thickness_m"] is not None
        assert abs(left["thickness_m"] - 0.3) < 1e-6

    def test_proximity_conflict_flagged_for_thin_boundaries(self):
        thin_surfaces = _sample_oriented_surfaces()
        thin_surfaces[1]["plane_point"] = [5.02, 0.0, 0.0]

        manifest = build_viewer_manifest(
            "job_1",
            _minimal_output_payload(),
            internal_boundaries_payload={"oriented_surfaces": thin_surfaces},
            proximity_threshold_m=0.05,
        )
        entities = manifest["internal_boundary_entities"]
        left = next(e for e in entities if e["surface_id"] == "ib_0_left_0")
        assert left["proximity_conflict"] is True
        assert left["thickness_m"] < 0.05

    def test_no_proximity_conflict_for_normal_thickness(self):
        manifest = build_viewer_manifest(
            "job_1",
            _minimal_output_payload(),
            internal_boundaries_payload={
                "oriented_surfaces": _sample_oriented_surfaces(),
            },
            proximity_threshold_m=0.05,
        )
        entities = manifest["internal_boundary_entities"]
        left = next(e for e in entities if e["surface_id"] == "ib_0_left_0")
        assert left["proximity_conflict"] is False

    def test_no_internal_boundaries_when_payload_is_none(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload())
        assert manifest["internal_boundary_entities"] == []
        assert manifest["layers"]["internal_boundaries"]["available"] is False
        assert manifest["layers"]["internal_boundaries"]["count"] == 0

    def test_summary_includes_internal_boundary_count(self):
        manifest = build_viewer_manifest(
            "job_1",
            _minimal_output_payload(),
            internal_boundaries_payload={
                "oriented_surfaces": _sample_oriented_surfaces(),
            },
        )
        assert manifest["summary"]["internal_boundary_count"] == 2

    def test_opening_integration_layer_available(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload())
        layer = manifest["layers"]["opening_integration"]
        assert layer["available"] is True
        assert layer["count"] == 1

    def test_min_area_threshold_in_manifest(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload(), min_area_threshold_m2=0.5)
        assert manifest["min_area_threshold_m2"] == 0.5

    def test_proximity_threshold_in_manifest(self):
        manifest = build_viewer_manifest("job_1", _minimal_output_payload(), proximity_threshold_m=0.1)
        assert manifest["proximity_threshold_m"] == 0.1
