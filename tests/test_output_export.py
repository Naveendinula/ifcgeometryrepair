from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from app.output_export import export_2lsb_obj, export_2lsb_xml


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
                "normal": [0, 0, 1],
                "centroid": [2, 1.5, 0],
                "area_m2": 1.5,
                "polygon_rings_3d": [
                    [[1.5, 1.0, 0], [2.5, 1.0, 0], [2.5, 2.0, 0], [1.5, 2.0, 0]]
                ],
                "triangles": [
                    [[1.5, 1.0, 0], [2.5, 1.0, 0], [2.5, 2.0, 0]],
                    [[1.5, 1.0, 0], [2.5, 2.0, 0], [1.5, 2.0, 0]],
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
                "normal": [1, 0, 0],
                "centroid": [5, 1.5, 1.5],
                "area_m2": 9.0,
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
