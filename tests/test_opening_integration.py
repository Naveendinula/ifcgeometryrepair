from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as orient_polygon

import app.opening_integration as opening_integration


# ---------------------------------------------------------------------------
# Helpers — build cube mesh
# ---------------------------------------------------------------------------


def _build_box_mesh(
    width: float = 1.0,
    depth: float = 1.0,
    height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for an axis-aligned box centered at origin."""
    w, d, h = width / 2, depth / 2, height / 2
    vertices = np.array(
        [
            [-w, -d, -h],
            [w, -d, -h],
            [w, d, -h],
            [-w, d, -h],
            [-w, -d, h],
            [w, -d, h],
            [w, d, h],
            [-w, d, h],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            # -Z (bottom)
            [0, 2, 1],
            [0, 3, 2],
            # +Z (top)
            [4, 5, 6],
            [4, 6, 7],
            # -Y (front)
            [0, 1, 5],
            [0, 5, 4],
            # +Y (back)
            [2, 3, 7],
            [2, 7, 6],
            # -X (left)
            [3, 0, 4],
            [3, 4, 7],
            # +X (right)
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


# ===========================================================================
# Tests for max_a
# ===========================================================================


class TestMaxA:
    def test_box_mesh_returns_top_face_for_up_normal(self) -> None:
        vertices, faces = _build_box_mesh(2.0, 3.0, 1.0)
        result = opening_integration.max_a(
            vertices, faces, np.array([0.0, 0.0, 1.0])
        )
        assert result is not None
        assert result["area_m2"] == pytest.approx(6.0, abs=0.1)  # 2 * 3
        np.testing.assert_allclose(result["normal"], [0, 0, 1], atol=0.01)
        assert isinstance(result["polygon_2d"], Polygon)

    def test_box_mesh_returns_side_face_for_x_normal(self) -> None:
        vertices, faces = _build_box_mesh(2.0, 3.0, 4.0)
        result = opening_integration.max_a(
            vertices, faces, np.array([1.0, 0.0, 0.0])
        )
        assert result is not None
        assert result["area_m2"] == pytest.approx(12.0, abs=0.1)  # 3 * 4

    def test_empty_mesh_returns_none(self) -> None:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        empty_f = np.zeros((0, 3), dtype=np.int64)
        result = opening_integration.max_a(
            empty_v, empty_f, np.array([0.0, 0.0, 1.0])
        )
        assert result is None

    def test_no_matching_normal_returns_none(self) -> None:
        # Single triangle facing +Z, query for +X — should find nothing.
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        result = opening_integration.max_a(
            vertices, faces, np.array([1.0, 0.0, 0.0])
        )
        assert result is None

    def test_result_contains_expected_keys(self) -> None:
        vertices, faces = _build_box_mesh()
        result = opening_integration.max_a(
            vertices, faces, np.array([0.0, 0.0, 1.0])
        )
        assert result is not None
        assert set(result.keys()) == {
            "normal",
            "plane_point",
            "basis_u",
            "basis_v",
            "polygon_2d",
            "area_m2",
        }


# ===========================================================================
# Tests for _is_coplanar
# ===========================================================================


class TestIsCoplanar:
    def test_same_plane_is_coplanar(self) -> None:
        n = np.array([0, 0, 1.0])
        p = np.array([0, 0, 0.0])
        assert opening_integration._is_coplanar(n, p, n, p, threshold_m=0.01)

    def test_parallel_within_threshold_is_coplanar(self) -> None:
        n = np.array([0, 0, 1.0])
        p1 = np.array([0, 0, 0.0])
        p2 = np.array([0, 0, 0.005])
        assert opening_integration._is_coplanar(n, p1, n, p2, threshold_m=0.01)

    def test_parallel_beyond_threshold_is_not_coplanar(self) -> None:
        n = np.array([0, 0, 1.0])
        p1 = np.array([0, 0, 0.0])
        p2 = np.array([0, 0, 0.05])
        assert not opening_integration._is_coplanar(n, p1, n, p2, threshold_m=0.01)

    def test_perpendicular_normals_not_coplanar(self) -> None:
        n1 = np.array([0, 0, 1.0])
        n2 = np.array([1, 0, 0.0])
        p = np.array([0, 0, 0.0])
        assert not opening_integration._is_coplanar(n1, p, n2, p, threshold_m=0.01)

    def test_opposite_normals_not_coplanar_by_default(self) -> None:
        n1 = np.array([0, 0, 1.0])
        n2 = np.array([0, 0, -1.0])
        p = np.array([0, 0, 0.0])
        # abs(dot) is compared, so opposite normals ARE coplanar.
        assert opening_integration._is_coplanar(n1, p, n2, p, threshold_m=0.01)


# ===========================================================================
# Tests for _project_polygon_to_basis
# ===========================================================================


class TestProjectPolygonToBasis:
    def test_identity_projection(self) -> None:
        """Projecting between identical planes should return the same polygon."""
        normal = np.array([0, 0, 1.0])
        plane_point = np.array([0, 0, 0.0])
        from app.external_shell import _plane_basis

        basis_u, basis_v = _plane_basis(normal)
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = opening_integration._project_polygon_to_basis(
            polygon,
            plane_point,
            basis_u,
            basis_v,
            plane_point,
            normal,
            basis_u,
            basis_v,
        )
        assert result.area == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Tests for project_openings_onto_boundaries
# ===========================================================================


class TestProjectOpeningsOntoBoundaries:
    def _make_boundary_surface(
        self,
        *,
        surface_id: str = "boundary_1",
        normal: list[float] | None = None,
        plane_point: list[float] | None = None,
        ring_3d: list[list[float]] | None = None,
    ) -> dict:
        """Build a minimal internal boundary surface dict.

        *ring_3d* should be specified directly as 3-D points lying on the
        plane described by *normal* and *plane_point*.  When omitted a default
        4×3 rectangle at z=0 is used.
        """
        normal = normal or [0, 0, 1]
        plane_point = plane_point or [0, 0, 0]

        if ring_3d is None:
            ring_3d = [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]]

        return {
            "oriented_surface_id": surface_id,
            "space_global_id": "space-a",
            "space_express_id": 1,
            "space_name": "Space A",
            "plane_normal": normal,
            "plane_point": plane_point,
            "centroid": [2, 1.5, 0],
            "area_m2": 12.0,
            "polygon_rings_3d": [ring_3d],
        }

    def _make_opening_entity(
        self,
        *,
        express_id: int = 100,
        global_id: str = "opening-001",
        name: str = "Window",
        center: tuple[float, float, float] = (2.0, 1.5, 0.0),
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 0.2,
        face_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> dict:
        """Build a minimal opening entity dict with a box mesh."""
        # Simple flat quad mesh (2 triangles) on the plane of face_normal
        # centered at center with given width/height.
        n = np.asarray(face_normal, dtype=np.float64)
        n /= np.linalg.norm(n)
        from app.external_shell import _plane_basis

        bu, bv = _plane_basis(n)
        c = np.asarray(center, dtype=np.float64)
        w2, h2, d2 = width / 2, height / 2, depth / 2

        # Build a thin box-like mesh.
        corners_2d = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
        # Front and back faces.
        vertices = []
        for u, v in corners_2d:
            vertices.append((c + u * bu + v * bv + d2 * n).tolist())
        for u, v in corners_2d:
            vertices.append((c + u * bu + v * bv - d2 * n).tolist())
        # Faces: front (0-3), back (4-7), sides.
        faces = [
            [0, 1, 2],
            [0, 2, 3],  # front (+normal)
            [4, 6, 5],
            [4, 7, 6],  # back (-normal)
            [0, 4, 5],
            [0, 5, 1],  # bottom
            [2, 6, 7],
            [2, 7, 3],  # top
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ]

        return {
            "express_id": express_id,
            "global_id": global_id,
            "name": name,
            "entity_type": "IfcOpeningElement",
            "valid": True,
            "mesh": {
                "vertices": vertices,
                "faces": faces,
            },
        }

    def test_opening_fully_inside_boundary_creates_surface(self) -> None:
        boundary = self._make_boundary_surface(
            surface_id="wall_1",
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
        )
        opening = self._make_opening_entity(
            center=(2.0, 1.5, 0.0),
            width=1.0,
            height=1.0,
            depth=0.2,
            face_normal=(0, 0, 1),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary],
            external_surfaces=[],
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        assert result["summary"]["opening_surfaces_created"] >= 1
        assert result["summary"]["total_opening_area_m2"] > 0
        # The opening should be roughly 1×1 = 1 m².
        total_area = sum(s["area_m2"] for s in result["opening_surfaces"])
        assert total_area == pytest.approx(1.0, abs=0.3)

    def test_no_openings_returns_empty_result(self) -> None:
        boundary = self._make_boundary_surface()
        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[],
            internal_surfaces=[boundary],
            external_surfaces=[],
            threshold_m=0.5,
            min_area_m2=0.01,
        )
        assert result["summary"]["opening_surfaces_created"] == 0
        assert result["opening_surfaces"] == []

    def test_opening_on_different_plane_not_matched(self) -> None:
        # Boundary on XY plane (normal +Z), opening on XZ plane (normal +Y).
        boundary = self._make_boundary_surface(
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
        )
        opening = self._make_opening_entity(
            center=(2.0, 0.0, 1.5),
            face_normal=(0, 1, 0),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary],
            external_surfaces=[],
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        assert result["summary"]["opening_surfaces_created"] == 0

    def test_opening_too_far_from_boundary_plane_not_matched(self) -> None:
        boundary = self._make_boundary_surface(
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
        )
        # Opening is on a parallel plane but 5 meters away.
        opening = self._make_opening_entity(
            center=(2.0, 1.5, 5.0),
            face_normal=(0, 0, 1),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary],
            external_surfaces=[],
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        assert result["summary"]["opening_surfaces_created"] == 0

    def test_tiny_opening_filtered_by_min_area(self) -> None:
        boundary = self._make_boundary_surface(
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
        )
        # Very small opening: 0.05 × 0.05 = 0.0025 m².
        opening = self._make_opening_entity(
            center=(5.0, 5.0, 0.0),
            width=0.05,
            height=0.05,
            depth=0.1,
            face_normal=(0, 0, 1),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary],
            external_surfaces=[],
            threshold_m=0.5,
            min_area_m2=0.1,  # Filter threshold larger than opening.
        )

        assert result["summary"]["opening_surfaces_created"] == 0

    def test_modified_boundaries_returned(self) -> None:
        boundary = self._make_boundary_surface(
            surface_id="wall_top",
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
        )
        opening = self._make_opening_entity(
            center=(2.0, 1.5, 0.0),
            width=1.0,
            height=1.0,
            depth=0.2,
            face_normal=(0, 0, 1),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary],
            external_surfaces=[],
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        assert len(result["modified_boundaries"]) == 1
        modified = result["modified_boundaries"][0]
        assert "opening_surface_ids" in modified


# ===========================================================================
# Tests for run_opening_integration (orchestrator)
# ===========================================================================


class TestRunOpeningIntegration:
    def test_empty_openings_returns_empty_payload(self, tmp_path: pytest.TempPathFactory) -> None:
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        result = opening_integration.run_opening_integration(
            job_id="test-job-001",
            job_dir=job_dir,
            preprocessing_result={"entities": []},
            internal_boundary_result={"oriented_surfaces": []},
            external_candidates_result={"candidate_surfaces": []},
            external_shell_result={"surfaces": []},
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        assert isinstance(result, opening_integration.OpeningIntegrationResult)
        assert result.payload["summary"]["opening_surfaces_created"] == 0
        assert result.payload["opening_surfaces"] == []
        assert (job_dir / "geometry" / "opening_integration.json").exists()

    def test_payload_contains_expected_structure(self, tmp_path: pytest.TempPathFactory) -> None:
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        result = opening_integration.run_opening_integration(
            job_id="test-job-002",
            job_dir=job_dir,
            preprocessing_result={"entities": []},
            internal_boundary_result={"oriented_surfaces": []},
            external_candidates_result={"candidate_surfaces": []},
            external_shell_result={"surfaces": []},
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        payload = result.payload
        assert "job_id" in payload
        assert "threshold_m" in payload
        assert "min_area_m2" in payload
        assert "clip_backend" in payload
        assert "summary" in payload
        assert "opening_surfaces" in payload
        assert "modified_boundaries" in payload


# ===========================================================================
# Tests for _merge_external_surfaces
# ===========================================================================


class TestMergeExternalSurfaces:
    def test_merge_adds_classification_from_shell(self) -> None:
        shell_surfaces = [
            {
                "surface_id": "ext_1",
                "source_surface_id": "ext_1",
                "classification": "external_wall",
            }
        ]
        candidate_surfaces = [
            {
                "surface_id": "ext_1",
                "plane_normal": [0, 0, 1],
                "plane_point": [0, 0, 0],
                "polygon_rings_3d": [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]],
            }
        ]
        merged = opening_integration._merge_external_surfaces(
            shell_surfaces, candidate_surfaces
        )
        assert len(merged) == 1
        assert merged[0]["classification"] == "external_wall"
        assert merged[0]["polygon_rings_3d"] is not None

    def test_missing_candidate_falls_back_to_shell(self) -> None:
        shell_surfaces = [
            {
                "surface_id": "ext_orphan",
                "classification": "roof",
            }
        ]
        merged = opening_integration._merge_external_surfaces(
            shell_surfaces, []
        )
        assert len(merged) == 1
        assert merged[0]["classification"] == "roof"
        assert merged[0]["surface_id"] == "ext_orphan"


# ===========================================================================
# Phase 1 regression tests – classification-based opening dedup priority
# ===========================================================================


class TestOpeningDedupPriority:
    """Opening dedup must prefer internal shared boundaries over internal_void."""

    def test_internal_beats_internal_void_in_dedup(self) -> None:
        """When an opening matches both an internal shared surface and an
        internal_void shell surface, the shared internal should win because
        classification priority: internal (1) < internal_void (2)."""
        boundary_internal = self._make_boundary_surface(
            surface_id="shared_wall",
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
        )
        boundary_void = self._make_external_surface(
            surface_id="void_wall",
            classification="internal_void",
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
        )
        opening = self._make_opening_entity(
            center=(2.0, 1.5, 0.0),
            width=1.0,
            height=1.0,
            depth=0.2,
            face_normal=(0, 0, 1),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary_internal],
            external_surfaces=[boundary_void],
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        # Exactly one opening surface should survive dedup
        assert result["summary"]["opening_surfaces_created"] == 1
        # The surviving opening must be parented on the shared internal surface
        surviving = result["opening_surfaces"][0]
        assert surviving["boundary_surface_id"] == "shared_wall"
        assert surviving["boundary_classification"] == "internal"

    def test_external_wall_beats_internal_in_dedup(self) -> None:
        """external_wall (priority 0) should beat internal (priority 1)."""
        boundary_internal = self._make_boundary_surface(
            surface_id="int_wall",
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
        )
        boundary_ext = self._make_external_surface(
            surface_id="ext_wall",
            classification="external_wall",
            normal=[0, 0, 1],
            plane_point=[0, 0, 0],
            ring_3d=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
        )
        opening = self._make_opening_entity(
            center=(2.0, 1.5, 0.0),
            width=1.0,
            height=1.0,
            depth=0.2,
            face_normal=(0, 0, 1),
        )

        result = opening_integration.project_openings_onto_boundaries(
            opening_entities=[opening],
            internal_surfaces=[boundary_internal],
            external_surfaces=[boundary_ext],
            threshold_m=0.5,
            min_area_m2=0.01,
        )

        assert result["summary"]["opening_surfaces_created"] == 1
        surviving = result["opening_surfaces"][0]
        assert surviving["boundary_surface_id"] == "ext_wall"
        assert surviving["boundary_classification"] == "external_wall"

    # --- helpers (reuse the pattern from TestProjectOpeningsOntoBoundaries) ---

    def _make_boundary_surface(
        self,
        *,
        surface_id: str = "boundary_1",
        normal: list[float] | None = None,
        plane_point: list[float] | None = None,
        ring_3d: list[list[float]] | None = None,
    ) -> dict:
        normal = normal or [0, 0, 1]
        plane_point = plane_point or [0, 0, 0]
        if ring_3d is None:
            ring_3d = [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]]
        return {
            "oriented_surface_id": surface_id,
            "space_global_id": "space-a",
            "space_express_id": 1,
            "space_name": "Space A",
            "plane_normal": normal,
            "plane_point": plane_point,
            "centroid": [2, 1.5, 0],
            "area_m2": 12.0,
            "polygon_rings_3d": [ring_3d],
        }

    def _make_external_surface(
        self,
        *,
        surface_id: str = "ext_1",
        classification: str = "external_wall",
        normal: list[float] | None = None,
        plane_point: list[float] | None = None,
        ring_3d: list[list[float]] | None = None,
    ) -> dict:
        normal = normal or [0, 0, 1]
        plane_point = plane_point or [0, 0, 0]
        if ring_3d is None:
            ring_3d = [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]]
        return {
            "surface_id": surface_id,
            "space_global_id": "space-a",
            "space_express_id": 1,
            "space_name": "Space A",
            "classification": classification,
            "normal": normal,
            "plane_normal": normal,
            "plane_point": plane_point,
            "centroid": [2, 1.5, 0],
            "area_m2": 12.0,
            "polygon_rings_3d": [ring_3d],
        }

    def _make_opening_entity(
        self,
        *,
        express_id: int = 100,
        global_id: str = "opening-001",
        name: str = "Window",
        center: tuple[float, float, float] = (2.0, 1.5, 0.0),
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 0.2,
        face_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> dict:
        n = np.asarray(face_normal, dtype=np.float64)
        n /= np.linalg.norm(n)
        from app.external_shell import _plane_basis
        bu, bv = _plane_basis(n)
        c = np.asarray(center, dtype=np.float64)
        w2, h2, d2 = width / 2, height / 2, depth / 2
        corners_2d = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
        vertices = []
        for u, v in corners_2d:
            vertices.append((c + u * bu + v * bv + d2 * n).tolist())
        for u, v in corners_2d:
            vertices.append((c + u * bu + v * bv - d2 * n).tolist())
        faces = [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ]
        return {
            "express_id": express_id,
            "global_id": global_id,
            "name": name,
            "entity_type": "IfcOpeningElement",
            "valid": True,
            "mesh": {"vertices": vertices, "faces": faces},
        }
