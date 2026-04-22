from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as orient_polygon

import app.internal_boundaries as internal_boundaries


def _make_planar_polygon(
    *,
    polygon_id: str,
    space_global_id: str,
    space_express_id: int,
    space_name: str,
    source_surface_id: str,
    plane_point: tuple[float, float, float],
    normal: tuple[float, float, float],
    shell: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> internal_boundaries.PlanarPolygon:
    normal_array = np.asarray(normal, dtype=np.float64)
    normal_array /= np.linalg.norm(normal_array)
    basis_u, basis_v = internal_boundaries._plane_basis(normal_array)
    polygon_2d = orient_polygon(Polygon(shell, holes or []), sign=1.0)
    return internal_boundaries.PlanarPolygon(
        polygon_id=polygon_id,
        space_global_id=space_global_id,
        space_express_id=space_express_id,
        space_name=space_name,
        source_surface_id=source_surface_id,
        normal=normal_array,
        plane_point=np.asarray(plane_point, dtype=np.float64),
        basis_u=basis_u,
        basis_v=basis_v,
        polygon_2d=polygon_2d,
    )


def _make_space(
    *,
    global_id: str,
    express_id: int,
    name: str,
    polygons: list[internal_boundaries.PlanarPolygon],
) -> internal_boundaries.SpaceGeometry:
    points: list[np.ndarray] = []
    for polygon in polygons:
        points.extend(
            internal_boundaries._polygon_ring_points_3d(
                polygon.polygon_2d,
                polygon.plane_point,
                polygon.basis_u,
                polygon.basis_v,
            )
        )
    point_array = np.asarray(points, dtype=np.float64)
    return internal_boundaries.SpaceGeometry(
        global_id=global_id,
        express_id=express_id,
        name=name,
        object_name=global_id,
        polygons=polygons,
        aabb_min=np.min(point_array, axis=0),
        aabb_max=np.max(point_array, axis=0),
    )


def _projection_result(
    *,
    left: internal_boundaries.PlanarPolygon,
    right: internal_boundaries.PlanarPolygon,
    raw_left_polygons: list[Polygon],
    raw_right_polygons: list[Polygon],
    raw_shared_polygons: list[Polygon],
    left_polygons: list[Polygon] | None = None,
    right_polygons: list[Polygon] | None = None,
    shared_polygons: list[Polygon] | None = None,
) -> internal_boundaries.IntersectionProjectionResult:
    normal, plane_point, basis_u, basis_v = internal_boundaries._midpoint_plane(left, right)
    return internal_boundaries.IntersectionProjectionResult(
        left_polygons=left_polygons if left_polygons is not None else raw_left_polygons,
        right_polygons=right_polygons if right_polygons is not None else raw_right_polygons,
        shared_polygons=shared_polygons if shared_polygons is not None else raw_shared_polygons,
        shared_normal=normal,
        shared_plane_point=plane_point,
        shared_basis_u=basis_u,
        shared_basis_v=basis_v,
        raw_left_polygons=raw_left_polygons,
        raw_right_polygons=raw_right_polygons,
        raw_shared_polygons=raw_shared_polygons,
    )


def test_intersection_projection_accepts_approximately_opposite_normals() -> None:
    left = _make_planar_polygon(
        polygon_id="left_poly",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )
    right = _make_planar_polygon(
        polygon_id="right_poly",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b",
        plane_point=(0.0, 0.0, 0.0005),
        normal=(0.002, 0.0, -0.999998),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )

    result = internal_boundaries.intersection_projection(left, right, threshold_m=0.01)

    assert result.left_polygons
    assert result.right_polygons
    assert result.shared_polygons
    assert result.left_polygons[0].area == pytest.approx(8.0, abs=0.05)


def test_intersection_projection_sets_preserve_polygon_holes() -> None:
    shell = [(-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)]
    hole = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    left = _make_planar_polygon(
        polygon_id="left_hole",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=shell,
        holes=[hole],
    )
    right = _make_planar_polygon(
        polygon_id="right_hole",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b",
        plane_point=(0.0, 0.0, 0.1),
        normal=(0.0, 0.0, -1.0),
        shell=shell,
        holes=[hole],
    )

    result = internal_boundaries.intersection_projection_sets(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[left]),
        _make_space(global_id="space-b", express_id=2, name="Space B", polygons=[right]),
        threshold_m=0.2,
        pair_index=0,
    )

    assert len(result["oriented_surfaces"]) == 2
    assert len(result["shared_surfaces"]) == 1
    assert len(result["oriented_surfaces"][0]["polygon_rings_3d"]) == 2
    assert len(result["shared_surfaces"][0]["polygon_rings_3d"]) == 2
    assert result["shared_surfaces"][0]["area_m2"] == pytest.approx(32.0, abs=0.05)


def test_intersection_projection_sets_emit_multiple_surfaces_for_one_space_pair() -> None:
    left = _make_planar_polygon(
        polygon_id="left_wide",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-4.0, -1.0), (4.0, -1.0), (4.0, 1.0), (-4.0, 1.0)],
    )
    right_one = _make_planar_polygon(
        polygon_id="right_one",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b_0",
        plane_point=(0.0, 0.0, 0.1),
        normal=(0.0, 0.0, -1.0),
        shell=[(-3.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (-3.0, 1.0)],
    )
    right_two = _make_planar_polygon(
        polygon_id="right_two",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b_1",
        plane_point=(0.0, 0.0, 0.1),
        normal=(0.0, 0.0, -1.0),
        shell=[(1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)],
    )

    result = internal_boundaries.intersection_projection_sets(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[left]),
        _make_space(global_id="space-b", express_id=2, name="Space B", polygons=[right_one, right_two]),
        threshold_m=0.2,
        pair_index=0,
    )

    assert len(result["oriented_surfaces"]) == 4
    assert len(result["shared_surfaces"]) == 2
    assert len(result["adjacency"]["oriented_surface_ids"]) == 4
    assert len(result["adjacency"]["shared_surface_ids"]) == 2
    assert result["adjacency"]["shared_area_m2"] == pytest.approx(8.0, abs=0.05)


def test_intersection_projection_rejects_non_opposite_normals() -> None:
    left = _make_planar_polygon(
        polygon_id="left_parallel",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )
    right = _make_planar_polygon(
        polygon_id="right_parallel",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b",
        plane_point=(0.0, 0.0, 0.05),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )

    result = internal_boundaries.intersection_projection(left, right, threshold_m=0.1)

    assert result.left_polygons == []
    assert result.right_polygons == []
    assert result.shared_polygons == []
    assert result.rejection_code == "non_opposite_normals"


def test_intersection_projection_rejects_pairs_beyond_threshold() -> None:
    left = _make_planar_polygon(
        polygon_id="left_far",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )
    right = _make_planar_polygon(
        polygon_id="right_far",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b",
        plane_point=(0.0, 0.0, 0.5),
        normal=(0.0, 0.0, -1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )

    result = internal_boundaries.intersection_projection(left, right, threshold_m=0.1)

    assert result.left_polygons == []
    assert result.right_polygons == []
    assert result.shared_polygons == []
    assert result.rejection_code == "pair_distance_exceeds_threshold"


def test_intersection_projection_sets_record_component_count_mismatch(monkeypatch) -> None:
    left = _make_planar_polygon(
        polygon_id="left_poly",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )
    right = _make_planar_polygon(
        polygon_id="right_poly",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b",
        plane_point=(0.0, 0.0, 0.1),
        normal=(0.0, 0.0, -1.0),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )
    left_overlap = orient_polygon(Polygon([(-2.0, -1.0), (0.0, -1.0), (0.0, 1.0), (-2.0, 1.0)]), sign=1.0)
    right_overlap = orient_polygon(Polygon([(-2.0, -1.0), (0.0, -1.0), (0.0, 1.0), (-2.0, 1.0)]), sign=1.0)

    def fake_projection(*_args, **_kwargs):
        return _projection_result(
            left=left,
            right=right,
            raw_left_polygons=[left_overlap],
            raw_right_polygons=[right_overlap],
            raw_shared_polygons=[],
            left_polygons=[left_overlap],
            right_polygons=[right_overlap],
            shared_polygons=[],
        )

    monkeypatch.setattr(internal_boundaries, "intersection_projection", fake_projection)

    result = internal_boundaries.intersection_projection_sets(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[left]),
        _make_space(global_id="space-b", express_id=2, name="Space B", polygons=[right]),
        threshold_m=0.2,
        pair_index=0,
    )

    assert len(result["oriented_surfaces"]) == 2
    assert result["shared_surfaces"] == []
    assert result["adjacency"]["shared_surface_ids"] == []
    assert result["adjacency"]["rejected_shared_component_count"] == 1
    assert result["rejected_shared_components"][0]["rejection_code"] == "component_count_mismatch"


def test_intersection_projection_sets_reject_degenerate_midpoint_overlap(monkeypatch) -> None:
    left = _make_planar_polygon(
        polygon_id="left_poly",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf_a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )
    right = _make_planar_polygon(
        polygon_id="right_poly",
        space_global_id="space-b",
        space_express_id=2,
        space_name="Space B",
        source_surface_id="surf_b",
        plane_point=(0.0, 0.0, 0.1),
        normal=(0.0, 0.0, -1.0),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )
    valid_overlap = orient_polygon(Polygon([(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)]), sign=1.0)
    tiny_shared = orient_polygon(Polygon([(0.0, 0.0), (0.04, 0.0), (0.04, 0.04), (0.0, 0.04)]), sign=1.0)

    def fake_projection(*_args, **_kwargs):
        return _projection_result(
            left=left,
            right=right,
            raw_left_polygons=[valid_overlap],
            raw_right_polygons=[valid_overlap],
            raw_shared_polygons=[tiny_shared],
            left_polygons=[valid_overlap],
            right_polygons=[valid_overlap],
            shared_polygons=[],
        )

    monkeypatch.setattr(internal_boundaries, "intersection_projection", fake_projection)

    result = internal_boundaries.intersection_projection_sets(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[left]),
        _make_space(global_id="space-b", express_id=2, name="Space B", polygons=[right]),
        threshold_m=0.2,
        pair_index=0,
    )

    assert len(result["oriented_surfaces"]) == 2
    assert result["shared_surfaces"] == []
    assert result["rejected_shared_components"][0]["rejection_code"] == "degenerate_midpoint_overlap"
