from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as orient_polygon

import app.external_candidates as external_candidates
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


def _make_internal_surface_ref(
    *,
    surface_id: str,
    plane_point: tuple[float, float, float],
    normal: tuple[float, float, float],
    shell: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> external_candidates.InternalSurfaceRef:
    normal_array = np.asarray(normal, dtype=np.float64)
    normal_array /= np.linalg.norm(normal_array)
    basis_u, basis_v = internal_boundaries._plane_basis(normal_array)
    polygon_2d = orient_polygon(Polygon(shell, holes or []), sign=1.0)
    return external_candidates.InternalSurfaceRef(
        surface_id=surface_id,
        space_global_id="space-a",
        space_express_id=1,
        source_surface_id="src",
        normal=normal_array,
        plane_point=np.asarray(plane_point, dtype=np.float64),
        basis_u=basis_u,
        basis_v=basis_v,
        polygon_2d=polygon_2d,
    )


def test_single_coplanar_subtractor_creates_hole_and_preserves_provenance() -> None:
    source = _make_planar_polygon(
        polygon_id="source",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf-a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-3.0, -2.0), (3.0, -2.0), (3.0, 2.0), (-3.0, 2.0)],
    )
    subtractor = _make_internal_surface_ref(
        surface_id="ibo_0",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )

    candidates, _, source_polygon_count, subtracted_count = external_candidates._subtract_internal_surfaces(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[source]),
        [subtractor],
        space_index=0,
    )

    assert source_polygon_count == 1
    assert subtracted_count == 1
    assert len(candidates) == 1
    assert candidates[0]["area_m2"] == pytest.approx(20.0, abs=0.05)
    assert len(candidates[0]["polygon_rings_3d"]) == 2
    assert candidates[0]["subtracted_internal_surface_ids"] == ["ibo_0"]


def test_multiple_coplanar_subtractors_produce_multiple_candidate_islands() -> None:
    source = _make_planar_polygon(
        polygon_id="source",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf-a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-5.0, -1.0), (5.0, -1.0), (5.0, 1.0), (-5.0, 1.0)],
    )
    subtractors = [
        _make_internal_surface_ref(
            surface_id="ibo_0",
            plane_point=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            shell=[(-3.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-3.0, 1.0)],
        ),
        _make_internal_surface_ref(
            surface_id="ibo_1",
            plane_point=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            shell=[(2.0, -1.0), (3.0, -1.0), (3.0, 1.0), (2.0, 1.0)],
        ),
    ]

    candidates, _, _, subtracted_count = external_candidates._subtract_internal_surfaces(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[source]),
        subtractors,
        space_index=0,
    )

    assert subtracted_count == 1
    assert len(candidates) == 3
    assert sum(candidate["area_m2"] for candidate in candidates) == pytest.approx(16.0, abs=0.05)


def test_nonparallel_internal_surface_is_not_subtracted() -> None:
    source = _make_planar_polygon(
        polygon_id="source",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf-a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
    )
    subtractor = _make_internal_surface_ref(
        surface_id="ibo_tilted",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.08, 0.0, 0.996795),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )

    candidates, _, _, subtracted_count = external_candidates._subtract_internal_surfaces(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[source]),
        [subtractor],
        space_index=0,
    )

    assert subtracted_count == 0
    assert len(candidates) == 1
    assert candidates[0]["area_m2"] == pytest.approx(16.0, abs=0.05)


def test_plane_offset_tolerance_blocks_noncoplanar_subtraction() -> None:
    source = _make_planar_polygon(
        polygon_id="source",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf-a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
    )
    subtractor = _make_internal_surface_ref(
        surface_id="ibo_far",
        plane_point=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )

    candidates, _, _, subtracted_count = external_candidates._subtract_internal_surfaces(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[source]),
        [subtractor],
        space_index=0,
    )

    assert subtracted_count == 0
    assert len(candidates) == 1
    assert candidates[0]["area_m2"] == pytest.approx(16.0, abs=0.05)


def test_candidate_triangles_preserve_owner_normal_direction() -> None:
    source = _make_planar_polygon(
        polygon_id="source",
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        source_surface_id="surf-a",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
    )

    candidates, _, _, subtracted_count = external_candidates._subtract_internal_surfaces(
        _make_space(global_id="space-a", express_id=1, name="Space A", polygons=[source]),
        [],
        space_index=0,
    )

    assert subtracted_count == 0
    assert len(candidates) == 1

    triangle = np.asarray(candidates[0]["triangles"][0], dtype=np.float64)
    derived_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    derived_normal /= np.linalg.norm(derived_normal)
    assert float(np.dot(derived_normal, source.normal)) > 0.99
