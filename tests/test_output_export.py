from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from app.gbxml_export import build_gbxml_preflight_payload
from app.output_export import export_2lsb_gbxml, export_2lsb_obj, export_2lsb_xml


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _empty_payloads() -> tuple[dict, dict, dict]:
    return (
        {  # opening_integration
            "summary": {
                "openings_processed": 0,
                "opening_surfaces_created": 0,
                "boundaries_with_openings": 0,
                "total_opening_area_m2": 0.0,
            },
            "opening_surfaces": [],
        },
        {  # internal_boundary
            "summary": {"oriented_surface_count": 0},
            "oriented_surfaces": [],
        },
        {  # external_shell
            "summary": {"candidate_surface_count": 0},
            "surfaces": [],
        },
    )


def _populated_payloads() -> tuple[dict, dict, dict]:
    opening_integration_payload = {
        "summary": {
            "openings_processed": 1,
            "opening_surfaces_created": 1,
            "boundaries_with_openings": 1,
            "total_opening_area_m2": 1.5,
        },
        "opening_surfaces": [
            {
                "surface_id": "oi_wall1_opening1",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "space_name": "Room A",
                "opening_express_id": 100,
                "opening_global_id": "opening-001",
                "classification": "opening",
                "boundary_surface_id": "ext_1",
                "boundary_classification": "external",
                "plane_normal": [1, 0, 0],
                "plane_point": [5, 1.5, 1.5],
                "normal": [0, 0, 1],
                "centroid": [2, 1.5, 0],
                "area_m2": 1.5,
                "polygon_rings_3d": [
                    [[5, 1.0, 1.0], [5, 2.0, 1.0], [5, 2.0, 2.5], [5, 1.0, 2.5]]
                ],
                "triangles": [
                    [[5, 1.0, 1.0], [5, 2.0, 1.0], [5, 2.0, 2.5]],
                    [[5, 1.0, 1.0], [5, 2.0, 2.5], [5, 1.0, 2.5]],
                ],
            }
        ],
    }
    internal_boundary_payload = {
        "summary": {"oriented_surface_count": 1},
        "oriented_surfaces": [
            {
                "oriented_surface_id": "os_1",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "space_name": "Room A",
                "plane_normal": [0, 0, 1],
                "centroid": [2, 1.5, 0],
                "area_m2": 12.0,
                "polygon_rings_3d": [
                    [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]]
                ],
                "triangles": [
                    [[0, 0, 0], [4, 0, 0], [4, 3, 0]],
                    [[0, 0, 0], [4, 3, 0], [0, 3, 0]],
                ],
            }
        ],
    }
    external_shell_payload = {
        "summary": {"candidate_surface_count": 1},
        "surfaces": [
            {
                "surface_id": "ext_1",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "external_wall",
                "plane_normal": [1, 0, 0],
                "plane_point": [5, 1.5, 1.5],
                "normal": [1, 0, 0],
                "centroid": [5, 1.5, 1.5],
                "area_m2": 9.0,
                "polygon_components_3d": [
                    [
                        [[5, 0, 0], [5, 3, 0], [5, 3, 3], [5, 0, 3]],
                    ]
                ],
                "triangles": [
                    [[5, 0, 0], [5, 3, 0], [5, 3, 3]],
                    [[5, 0, 0], [5, 3, 3], [5, 0, 3]],
                ],
            }
        ],
    }
    return opening_integration_payload, internal_boundary_payload, external_shell_payload


# ===========================================================================
# Tests for XML export
# ===========================================================================


class TestExport2lsbXml:
    def test_empty_payloads_produce_valid_xml(self, tmp_path) -> None:
        oi, ib, es = _empty_payloads()
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path)
        assert path.exists()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "SecondLevelSpaceBoundaries"
        assert root.find("InternalBoundaries") is not None
        assert root.find("ExternalBoundaries") is not None
        assert root.find("OpeningSurfaces") is not None
        assert root.find("Summary") is not None

    def test_populated_payloads_produce_expected_structure(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path)

        tree = ET.parse(path)
        root = tree.getroot()

        internal_surfaces = root.findall("InternalBoundaries/Surface")
        assert len(internal_surfaces) == 1
        assert internal_surfaces[0].get("id") == "os_1"

        external_surfaces = root.findall("ExternalBoundaries/Surface")
        assert len(external_surfaces) == 1

        opening_surfaces = root.findall("OpeningSurfaces/Surface")
        assert len(opening_surfaces) == 1
        assert opening_surfaces[0].get("id") == "oi_wall1_opening1"

    def test_summary_counts_are_correct(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path)

        tree = ET.parse(path)
        summary = tree.getroot().find("Summary")
        assert summary.findtext("InternalSurfaceCount") == "1"
        assert summary.findtext("ExternalSurfaceCount") == "1"
        assert summary.findtext("OpeningSurfaceCount") == "1"

    def test_surface_element_contains_geometry(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path)

        tree = ET.parse(path)
        surface = tree.getroot().find("InternalBoundaries/Surface")
        assert surface.find("Normal") is not None
        assert surface.find("Geometry") is not None

    def test_returns_output_path(self, tmp_path) -> None:
        oi, ib, es = _empty_payloads()
        path = tmp_path / "output.xml"
        result = export_2lsb_xml(oi, ib, es, path)
        assert result == path


# ===========================================================================
# Tests for OBJ export
# ===========================================================================


class TestExport2lsbObj:
    def test_empty_payloads_produce_empty_obj(self, tmp_path) -> None:
        oi, ib, es = _empty_payloads()
        path = tmp_path / "output.obj"
        export_2lsb_obj(oi, ib, es, path)
        assert path.exists()

    def test_populated_payloads_produce_obj_with_groups(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        path = tmp_path / "output.obj"
        export_2lsb_obj(oi, ib, es, path)

        content = path.read_text(encoding="utf-8")
        assert "v " in content  # has vertices
        assert "f " in content  # has faces

    def test_returns_output_path(self, tmp_path) -> None:
        oi, ib, es = _empty_payloads()
        path = tmp_path / "output.obj"
        result = export_2lsb_obj(oi, ib, es, path)
        assert result == path

    def test_creates_parent_directory(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        path = tmp_path / "nested" / "dir" / "output.obj"
        export_2lsb_obj(oi, ib, es, path)
        assert path.exists()


# ===========================================================================
# Payloads with shared surfaces (for gbXML middle-plane tests)
# ===========================================================================


def _populated_payloads_with_shared() -> tuple[dict, dict, dict]:
    """Return payloads that include shared_surfaces and modified_boundaries."""
    oi, ib, es = _populated_payloads()

    # Add shared surfaces (middle-plane representation) used by gbXML export.
    ib["shared_surfaces"] = [
        {
            "shared_surface_id": "ib_0_0",
            "space_a_global_id": "space-a",
            "space_a_express_id": 1,
            "space_b_global_id": "space-b",
            "space_b_express_id": 2,
            "oriented_surface_ids": ["ibo_0_0_a", "ibo_0_0_b"],
            "area_m2": 8.4,
            "plane_normal": [1, 0, 0],
            "plane_point": [4, 1.5, 1.5],
            "centroid": [4, 1.5, 1.5],
            "polygon_rings_3d": [
                [[4, 0, 0], [4, 3, 0], [4, 3, 3], [4, 0, 3]]
            ],
        }
    ]

    # Simulate opening subtraction on the external boundary surface.
    oi["modified_boundaries"] = [
        {
            "surface_id": "ext_1",
            "boundary_type": "external",
            "opening_surface_ids": ["oi_wall1_opening1"],
            "remainder_polygon_rings_3d": [
                [[5, 0, 0], [5, 3, 0], [5, 3, 2], [5, 0, 2]]
            ],
            "remainder_area_m2": 7.5,
        }
    ]

    return oi, ib, es


def _preprocessing_result(*spaces: dict) -> dict:
    return {"entities": list(spaces)}


def _space_entity(
    space_id: str,
    express_id: int,
    name: str,
    *,
    valid: bool = True,
    closed: bool = True,
    manifold: bool = True,
    volume_m3: float = 30.0,
) -> dict:
    return {
        "entity_type": "IfcSpace",
        "global_id": space_id,
        "express_id": express_id,
        "name": name,
        "valid": valid,
        "closed": closed,
        "manifold": manifold,
        "volume_m3": volume_m3,
    }


# ===========================================================================
# Tests for gbXML export (Paper Section 3.4)
# ===========================================================================


NS = {"gb": "http://www.gbxml.org/schema"}


class TestExport2lsbGbxml:
    def test_empty_payloads_produce_valid_gbxml(self, tmp_path) -> None:
        oi, ib, es = _empty_payloads()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(oi, ib, es, path)
        assert path.exists()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag.endswith("gbXML")
        assert root.get("lengthUnit") == "Meters"
        assert root.find("gb:Campus", NS) is not None
        assert root.find("gb:Campus/gb:Building", NS) is not None

    def test_populated_payloads_produce_surfaces(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        surfaces = root.findall("gb:Campus/gb:Surface", NS)
        # Should have: 1 shared (middle-plane) + 1 external = 2 surfaces
        assert len(surfaces) >= 2

    def test_shared_surfaces_use_middle_plane(self, tmp_path) -> None:
        """gbXML should use shared/middle-plane surfaces instead of oriented pairs."""
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        surface_ids = [s.get("id") for s in root.findall("gb:Campus/gb:Surface", NS)]
        assert "ib_0_0" in surface_ids
        assert "ibo_0_0_a" not in surface_ids
        assert "ibo_0_0_b" not in surface_ids

    def test_shared_surfaces_reference_both_adjacent_spaces(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        shared_surface = next(
            surface for surface in root.findall("gb:Campus/gb:Surface", NS) if surface.get("id") == "ib_0_0"
        )
        adjacent_refs = [node.get("spaceIdRef") for node in shared_surface.findall("gb:AdjacentSpaceId", NS)]
        assert adjacent_refs == ["space-a", "space-b"]

    def test_external_surface_has_correct_type(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        for surface in root.findall("gb:Campus/gb:Surface", NS):
            if surface.get("id") == "ext_1":
                assert surface.get("surfaceType") == "ExteriorWall"

    def test_opening_attached_to_parent(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        for surface in root.findall("gb:Campus/gb:Surface", NS):
            if surface.get("id") == "ext_1":
                openings = surface.findall("gb:Opening", NS)
                assert len(openings) == 1

    def test_opening_parent_surface_uses_original_boundary_not_remainder(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        ext_surface = next(
            surface for surface in root.findall("gb:Campus/gb:Surface", NS) if surface.get("id") == "ext_1"
        )
        coordinates = [
            float(point.findtext("gb:Coordinate[3]", namespaces=NS))
            for point in ext_surface.findall("gb:PlanarGeometry/gb:PolyLoop/gb:CartesianPoint", NS)
        ]
        assert max(coordinates) == pytest.approx(3.0)

    def test_multiple_openings_attach_to_same_parent(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        oi["opening_surfaces"].append(
            {
                "surface_id": "oi_wall1_opening2",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "space_name": "Room A",
                "opening_express_id": 101,
                "opening_global_id": "opening-002",
                "classification": "opening",
                "boundary_surface_id": "ext_1",
                "boundary_classification": "external",
                "plane_normal": [1, 0, 0],
                "plane_point": [5, 2.4, 1.5],
                "normal": [1, 0, 0],
                "centroid": [5, 2.4, 1.5],
                "area_m2": 0.5,
                "polygon_rings_3d": [
                    [[5, 2.1, 1.0], [5, 2.6, 1.0], [5, 2.6, 2.0], [5, 2.1, 2.0]]
                ],
                "triangles": [
                    [[5, 2.1, 1.0], [5, 2.6, 1.0], [5, 2.6, 2.0]],
                    [[5, 2.1, 1.0], [5, 2.6, 2.0], [5, 2.1, 2.0]],
                ],
            }
        )
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            preprocessing_result=_preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        ext_surface = next(
            surface for surface in root.findall("gb:Campus/gb:Surface", NS) if surface.get("id") == "ext_1"
        )
        assert len(ext_surface.findall("gb:Opening", NS)) == 2

    def test_returns_output_path(self, tmp_path) -> None:
        oi, ib, es = _empty_payloads()
        path = tmp_path / "output.gbxml"
        result = export_2lsb_gbxml(oi, ib, es, path)
        assert result == path


class TestGbxmlPreflight:
    def test_disconnected_surface_splits_into_multiple_surfaces(self) -> None:
        oi, ib, es = _empty_payloads()
        es["surfaces"] = [
            {
                "surface_id": "ext_split",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "external_wall",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "normal": [0, 0, 1],
                "centroid": [0, 0, 0],
                "polygon_components_3d": [
                    [[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]],
                    [[[3, 0, 0], [5, 0, 0], [5, 2, 0], [3, 2, 0]]],
                ],
                "area_m2": 8.0,
            }
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        assert payload["status"] == "warning"
        assert [surface["id"] for surface in payload["export_plan"]["surfaces"]] == [
            "ext_split__part_1",
            "ext_split__part_2",
        ]
        assert any(issue["code"] == "surface_split_into_components" for issue in payload["warnings"])

    def test_zero_volume_space_is_omitted(self) -> None:
        oi, ib, es = _empty_payloads()
        es["surfaces"] = [
            {
                "surface_id": "ext_invalid",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "external_wall",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "normal": [0, 0, 1],
                "centroid": [0, 0, 0],
                "polygon_components_3d": [
                    [[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]],
                ],
                "area_m2": 4.0,
            }
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A", volume_m3=0.0)),
            ib,
            es,
            oi,
        )

        assert payload["status"] == "invalid"
        assert payload["export_plan"]["surfaces"] == []
        assert any(issue["code"] == "invalid_export_space" for issue in payload["blockers"])
        assert any(item["id"] == "ext_invalid" for item in payload["omitted_entities"])

    def test_opening_outside_parent_becomes_blocker_and_omitted(self) -> None:
        oi, ib, es = _populated_payloads()
        oi["opening_surfaces"][0]["polygon_rings_3d"] = [
            [[5, 10.0, 1.0], [5, 11.0, 1.0], [5, 11.0, 2.0], [5, 10.0, 2.0]]
        ]
        oi["opening_surfaces"][0]["triangles"] = [
            [[5, 10.0, 1.0], [5, 11.0, 1.0], [5, 11.0, 2.0]],
            [[5, 10.0, 1.0], [5, 11.0, 2.0], [5, 10.0, 2.0]],
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        assert payload["status"] == "invalid"
        assert any(issue["code"] == "opening_not_bounded_by_parent" for issue in payload["blockers"])
        assert any(item["id"] == "oi_wall1_opening1" for item in payload["omitted_entities"])

    def test_non_coplanar_opening_becomes_blocker_and_omitted(self) -> None:
        oi, ib, es = _populated_payloads()
        oi["opening_surfaces"][0]["plane_point"] = [4.7, 1.5, 1.5]
        oi["opening_surfaces"][0]["polygon_rings_3d"] = [
            [[4.7, 1.0, 1.0], [4.7, 2.0, 1.0], [4.7, 2.0, 2.5], [4.7, 1.0, 2.5]]
        ]
        oi["opening_surfaces"][0]["triangles"] = [
            [[4.7, 1.0, 1.0], [4.7, 2.0, 1.0], [4.7, 2.0, 2.5]],
            [[4.7, 1.0, 1.0], [4.7, 2.0, 2.5], [4.7, 1.0, 2.5]],
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        assert payload["status"] == "invalid"
        assert any(issue["code"] == "opening_not_coplanar_with_parent" for issue in payload["blockers"])
        assert any(item["id"] == "oi_wall1_opening1" for item in payload["omitted_entities"])

    def test_winding_correction_is_reported(self) -> None:
        oi, ib, es = _empty_payloads()
        es["surfaces"] = [
            {
                "surface_id": "ext_clockwise",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "external_wall",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "normal": [0, 0, 1],
                "centroid": [0, 0, 0],
                "polygon_components_3d": [
                    [[[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0]]],
                ],
                "area_m2": 4.0,
            }
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        assert payload["status"] == "warning"
        assert any(issue["code"] == "winding_corrected" for issue in payload["warnings"])

    def test_rejected_internal_midpoint_pair_becomes_blocker(self) -> None:
        oi, ib, es = _empty_payloads()
        ib["summary"] = {"oriented_surface_count": 2, "shared_surface_count": 0}
        ib["oriented_surfaces"] = [
            {
                "oriented_surface_id": "ibo_0_0_a",
                "shared_surface_id": None,
                "space_global_id": "space-a",
                "space_express_id": 1,
            },
            {
                "oriented_surface_id": "ibo_0_0_b",
                "shared_surface_id": None,
                "space_global_id": "space-b",
                "space_express_id": 2,
            },
        ]
        ib["adjacencies"] = [
            {
                "space_a_global_id": "space-a",
                "space_a_express_id": 1,
                "space_b_global_id": "space-b",
                "space_b_express_id": 2,
                "oriented_surface_ids": ["ibo_0_0_a", "ibo_0_0_b"],
                "shared_surface_ids": [],
                "rejected_shared_surface_ids": ["ib_0_0"],
                "rejected_shared_component_count": 1,
                "shared_area_m2": 0.0,
            }
        ]
        ib["rejected_shared_components"] = [
            {
                "shared_surface_id": "ib_0_0",
                "space_a_global_id": "space-a",
                "space_a_express_id": 1,
                "space_b_global_id": "space-b",
                "space_b_express_id": 2,
                "source_surface_a_id": "surf_a",
                "source_surface_b_id": "surf_b",
                "source_polygon_a_id": "poly_a",
                "source_polygon_b_id": "poly_b",
                "oriented_surface_ids": ["ibo_0_0_a", "ibo_0_0_b"],
                "rejection_code": "component_count_mismatch",
                "rejection_message": "Synthetic midpoint mismatch.",
            }
        ]
        es["surfaces"] = [
            {
                "surface_id": "ext_1",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "external_wall",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "normal": [0, 0, 1],
                "centroid": [0, 0, 0],
                "polygon_components_3d": [
                    [[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]],
                ],
                "area_m2": 4.0,
            }
        ]

        payload = build_gbxml_preflight_payload(
            _preprocessing_result(
                _space_entity("space-a", 1, "Room A"),
                _space_entity("space-b", 2, "Room B"),
            ),
            ib,
            es,
            oi,
        )

        assert payload["status"] == "invalid"
        assert any(issue["code"] == "internal_midpoint_component_count_mismatch" for issue in payload["blockers"])
        midpoint_blocker = next(
            issue for issue in payload["blockers"] if issue["code"] == "internal_midpoint_component_count_mismatch"
        )
        assert midpoint_blocker["entity"]["surface_id"] == "ib_0_0"
        assert midpoint_blocker["entity"]["source_surface_ids"] == ["surf_a", "surf_b"]
        assert midpoint_blocker["entity"]["source_polygon_ids"] == ["poly_a", "poly_b"]
        assert all(surface["id"] != "ib_0_0" for surface in payload["export_plan"]["surfaces"])
        assert any(item["id"] == "ib_0_0" for item in payload["omitted_entities"])

# ===========================================================================
# Tests for min area filtering (Paper Section 9.2)
# ===========================================================================


class TestMinAreaFiltering:
    def test_xml_excludes_tiny_surfaces(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        # Set the internal surface area below threshold.
        ib["oriented_surfaces"][0]["area_m2"] = 0.1
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path, min_area_m2=0.25)

        tree = ET.parse(path)
        root = tree.getroot()
        internal_surfaces = root.findall("InternalBoundaries/Surface")
        assert len(internal_surfaces) == 0  # Excluded by area filter

    def test_xml_keeps_large_surfaces(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path, min_area_m2=0.25)

        tree = ET.parse(path)
        root = tree.getroot()
        internal_surfaces = root.findall("InternalBoundaries/Surface")
        assert len(internal_surfaces) == 1  # 12.0 m² > 0.25

    def test_obj_excludes_tiny_surfaces(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        ib["oriented_surfaces"][0]["area_m2"] = 0.1
        es["surfaces"][0]["area_m2"] = 0.05
        path = tmp_path / "output.obj"
        export_2lsb_obj(oi, ib, es, path, min_area_m2=0.25)

        content = path.read_text(encoding="utf-8")
        # Only the opening surface (1.5 m²) should remain.
        assert "opening_" in content
        assert "internal_" not in content
        assert "external_wall_" not in content

    def test_gbxml_keeps_surface_when_min_area_is_zero(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads()
        es["surfaces"][0]["area_m2"] = 0.05
        path = tmp_path / "output.gbxml"
        export_2lsb_gbxml(
            oi,
            ib,
            es,
            path,
            min_area_m2=0.0,
            preprocessing_result=_preprocessing_result(_space_entity("space-a", 1, "Room A")),
        )

        tree = ET.parse(path)
        root = tree.getroot()
        surface_ids = [surface.get("id") for surface in root.findall("gb:Campus/gb:Surface", NS)]
        assert "ext_1" in surface_ids


# ===========================================================================
# Tests for opening-subtracted boundaries (Paper Section 3.3.4)
# ===========================================================================


class TestOpeningSubtraction:
    def test_xml_exports_subtracted_area(self, tmp_path) -> None:
        oi, ib, es = _populated_payloads_with_shared()
        path = tmp_path / "output.xml"
        export_2lsb_xml(oi, ib, es, path, min_area_m2=0.0)

        tree = ET.parse(path)
        root = tree.getroot()
        for surface in root.findall("ExternalBoundaries/Surface"):
            if surface.get("id") == "ext_1":
                area = float(surface.findtext("AreaM2", "0"))
                # Should reflect remainder_area_m2 = 7.5, not original 9.0
                assert area == 7.5


# ===========================================================================
# Phase 1 regression tests – classification exclusion & opening orphan policy
# ===========================================================================


class TestInternalVoidNotExcluded:
    """internal_void surfaces must appear in the export as InteriorWall."""

    def test_internal_void_surfaces_exported_as_interior_wall(self) -> None:
        oi, ib, es = _empty_payloads()
        es["surfaces"] = [
            {
                "surface_id": "void_1",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "internal_void",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "normal": [0, 0, 1],
                "centroid": [1, 1, 0],
                "polygon_components_3d": [
                    [[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]],
                ],
                "area_m2": 4.0,
            }
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        surfaces = payload["export_plan"]["surfaces"]
        assert len(surfaces) == 1
        assert surfaces[0]["surface_type"] == "InteriorWall"
        # Must NOT be omitted
        assert not any(item["id"] == "void_1" for item in payload["omitted_entities"])

    def test_unclassified_surfaces_still_excluded(self) -> None:
        oi, ib, es = _empty_payloads()
        es["surfaces"] = [
            {
                "surface_id": "unc_1",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "unclassified",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "normal": [0, 0, 1],
                "centroid": [1, 1, 0],
                "polygon_components_3d": [
                    [[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]],
                ],
                "area_m2": 4.0,
            }
        ]
        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        assert payload["export_plan"]["surfaces"] == []
        assert any(item["id"] == "unc_1" for item in payload["omitted_entities"])


class TestOpeningOrphanPolicy:
    """Opening whose parent was excluded by classification should warn, not block."""

    def test_opening_on_excluded_parent_is_warning_not_blocker(self) -> None:
        oi, ib, es = _empty_payloads()
        # Parent surface is unclassified → excluded
        es["surfaces"] = [
            {
                "surface_id": "excl_parent",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "classification": "unclassified",
                "plane_normal": [1, 0, 0],
                "plane_point": [5, 0, 0],
                "normal": [1, 0, 0],
                "centroid": [5, 1.5, 1.5],
                "polygon_components_3d": [
                    [[[5, 0, 0], [5, 3, 0], [5, 3, 3], [5, 0, 3]]],
                ],
                "area_m2": 9.0,
            }
        ]
        oi["opening_surfaces"] = [
            {
                "surface_id": "orphan_opening",
                "space_global_id": "space-a",
                "space_express_id": 1,
                "opening_express_id": 200,
                "opening_global_id": "opening-orphan",
                "classification": "opening",
                "boundary_surface_id": "excl_parent",
                "boundary_classification": "unclassified",
                "plane_normal": [1, 0, 0],
                "plane_point": [5, 1.5, 1.5],
                "normal": [1, 0, 0],
                "centroid": [5, 1.5, 1.5],
                "area_m2": 1.0,
                "polygon_rings_3d": [
                    [[5, 1.0, 1.0], [5, 2.0, 1.0], [5, 2.0, 2.0], [5, 1.0, 2.0]]
                ],
                "triangles": [
                    [[5, 1.0, 1.0], [5, 2.0, 1.0], [5, 2.0, 2.0]],
                    [[5, 1.0, 1.0], [5, 2.0, 2.0], [5, 1.0, 2.0]],
                ],
            }
        ]

        payload = build_gbxml_preflight_payload(
            _preprocessing_result(_space_entity("space-a", 1, "Room A")),
            ib,
            es,
            oi,
        )

        # Opening should produce a warning, NOT a blocker
        assert any(w["code"] == "opening_parent_excluded" for w in payload["warnings"])
        assert not any(b["code"] == "opening_missing_parent_surface" for b in payload["blockers"])
        assert any(item["id"] == "orphan_opening" for item in payload["omitted_entities"])
