from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

import app.external_shell as external_shell


def _make_surface_patch(
    *,
    surface_id: str,
    plane_point: tuple[float, float, float],
    normal: tuple[float, float, float],
    shell: list[tuple[float, float]],
) -> external_shell.SurfacePatch:
    normal_array = np.asarray(normal, dtype=np.float64)
    normal_array /= np.linalg.norm(normal_array)
    plane_point_array = np.asarray(plane_point, dtype=np.float64)
    basis_u, basis_v = external_shell._plane_basis(normal_array)
    polygon_2d = Polygon(shell)
    triangles_3d = external_shell._triangulate_polygons_3d([polygon_2d], plane_point_array, basis_u, basis_v)
    mesh = external_shell._triangles_to_mesh(triangles_3d)
    centroid = np.asarray(
        external_shell._lift_point(polygon_2d.centroid.coords[0], plane_point_array, basis_u, basis_v),
        dtype=np.float64,
    )
    return external_shell.SurfacePatch(
        surface_id=surface_id,
        object_name=surface_id,
        space_global_id="space-a",
        space_express_id=1,
        space_name="Space A",
        classification="unclassified",
        area_m2=float(polygon_2d.area),
        normal=normal_array,
        centroid=centroid,
        plane_point=plane_point_array,
        basis_u=basis_u,
        basis_v=basis_v,
        union_polygon_2d=polygon_2d,
        triangles_3d=[np.asarray(triangle, dtype=np.float64) for triangle in triangles_3d],
        mesh=mesh,
    )


def test_alpha_wrap_hit_classifies_surface_as_external() -> None:
    surface = _make_surface_patch(
        surface_id="surf_0",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )
    shell_mesh = {
        "vertices": [[-2.0, -2.0, 0.01], [2.0, -2.0, 0.01], [2.0, 2.0, 0.01], [-2.0, 2.0, 0.01]],
        "faces": [[0, 1, 2], [0, 2, 3]],
    }

    shell_match_count, internal_void_count, unclassified_count = external_shell._classify_with_alpha_wrap(
        [surface],
        shell_mesh,
        offset_tolerance_m=0.02,
    )

    assert shell_match_count == 1
    assert internal_void_count == 0
    assert unclassified_count == 0
    assert surface.classification == "roof"
    assert surface.reason.startswith("alpha_wrap_hit:")


def test_alpha_wrap_miss_classifies_surface_as_internal_void() -> None:
    surface = _make_surface_patch(
        surface_id="surf_0",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )
    shell_mesh = {
        "vertices": [[5.0, 5.0, 0.01], [7.0, 5.0, 0.01], [7.0, 7.0, 0.01], [5.0, 7.0, 0.01]],
        "faces": [[0, 1, 2], [0, 2, 3]],
    }

    shell_match_count, internal_void_count, unclassified_count = external_shell._classify_with_alpha_wrap(
        [surface],
        shell_mesh,
        offset_tolerance_m=0.02,
    )

    assert shell_match_count == 0
    assert internal_void_count == 1
    assert unclassified_count == 0
    assert surface.classification == "internal_void"
    assert surface.reason == "no_alpha_wrap_hit"


def test_triangle_aabb_tree_matches_bruteforce_query_results() -> None:
    triangles = [
        external_shell.WrapTriangle(
            triangle_id=f"aw_{index}",
            vertices=np.asarray(
                [[index * 3.0, 0.0, 0.0], [index * 3.0 + 1.0, 0.0, 0.0], [index * 3.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            plane_point=np.asarray([index * 3.0, 0.0, 0.0], dtype=np.float64),
            aabb_min=np.asarray([index * 3.0, 0.0, 0.0], dtype=np.float64),
            aabb_max=np.asarray([index * 3.0 + 1.0, 1.0, 0.0], dtype=np.float64),
        )
        for index in range(6)
    ]

    tree = external_shell._build_triangle_aabb_tree(triangles)
    query_min = np.asarray([4.4, -0.5, -0.5], dtype=np.float64)
    query_max = np.asarray([7.2, 1.5, 0.5], dtype=np.float64)

    indexed_ids = {
        triangle.triangle_id
        for triangle in external_shell._query_triangle_aabb_tree(tree, query_min, query_max)
    }
    brute_force_ids = {
        triangle.triangle_id
        for triangle in triangles
        if external_shell._aabb_overlap(triangle.aabb_min, triangle.aabb_max, query_min, query_max)
    }

    assert indexed_ids == brute_force_ids
    assert indexed_ids == {"aw_2"}


def test_alpha_wrap_match_requires_triangle_to_stay_within_offset_tolerance() -> None:
    surface = _make_surface_patch(
        surface_id="surf_0",
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        shell=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    )
    shell_mesh = {
        "vertices": [[-2.0, -2.0, 0.20], [2.0, -2.0, 0.20], [2.0, 2.0, 0.20], [-2.0, 2.0, 0.20]],
        "faces": [[0, 1, 2], [0, 2, 3]],
    }

    shell_match_count, internal_void_count, _ = external_shell._classify_with_alpha_wrap(
        [surface],
        shell_mesh,
        offset_tolerance_m=0.02,
    )

    assert shell_match_count == 0
    assert internal_void_count == 1
