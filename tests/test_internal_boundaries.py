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
