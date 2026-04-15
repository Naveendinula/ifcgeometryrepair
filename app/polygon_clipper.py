from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

try:
    import pyclipper  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised when the optional dependency is installed.
    pyclipper = None


CLIPPER_SCALE = 1_000_000.0
BACKEND_NAME = "pyclipper" if pyclipper is not None else "shapely"


def intersection(left: Polygon | MultiPolygon, right: Polygon | MultiPolygon) -> list[Polygon]:
    if pyclipper is not None:
        try:
            return _intersection_pyclipper(left, right)
        except Exception:
            pass
    return _extract_polygons(left.intersection(right))


def difference(subject: Polygon | MultiPolygon, clip: Polygon | MultiPolygon) -> list[Polygon]:
    if pyclipper is not None:
        try:
            return _difference_pyclipper(subject, clip)
        except Exception:
            pass
    return _extract_polygons(subject.difference(clip))


def union(polygons: Iterable[Polygon | MultiPolygon]) -> list[Polygon]:
    geometries = [geometry for geometry in polygons if geometry is not None]
    if not geometries:
        return []
    if pyclipper is not None:
        try:
            return _union_pyclipper(geometries)
        except Exception:
            pass
    return _extract_polygons(unary_union(geometries))


def _intersection_pyclipper(left: Polygon | MultiPolygon, right: Polygon | MultiPolygon) -> list[Polygon]:
    pc = pyclipper.Pyclipper()
    subject_paths = _geometry_to_paths(left)
    clip_paths = _geometry_to_paths(right)
    if not subject_paths or not clip_paths:
        return []
    pc.AddPaths(subject_paths, pyclipper.PT_SUBJECT, True)
    pc.AddPaths(clip_paths, pyclipper.PT_CLIP, True)
    tree = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    return _polytree_to_polygons(tree)


def _difference_pyclipper(subject: Polygon | MultiPolygon, clip: Polygon | MultiPolygon) -> list[Polygon]:
    pc = pyclipper.Pyclipper()
    subject_paths = _geometry_to_paths(subject)
    clip_paths = _geometry_to_paths(clip)
    if not subject_paths:
        return []
    pc.AddPaths(subject_paths, pyclipper.PT_SUBJECT, True)
    if clip_paths:
        pc.AddPaths(clip_paths, pyclipper.PT_CLIP, True)
    tree = pc.Execute2(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    return _polytree_to_polygons(tree)


def _union_pyclipper(polygons: list[Polygon | MultiPolygon]) -> list[Polygon]:
    pc = pyclipper.Pyclipper()
    paths: list[list[tuple[int, int]]] = []
    for geometry in polygons:
        paths.extend(_geometry_to_paths(geometry))
    if not paths:
        return []
    pc.AddPaths(paths, pyclipper.PT_SUBJECT, True)
    tree = pc.Execute2(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    return _polytree_to_polygons(tree)


def _geometry_to_paths(geometry: Polygon | MultiPolygon) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = []
    for polygon in _extract_polygons(geometry):
        shell = _ring_to_path(list(polygon.exterior.coords)[:-1])
        if len(shell) >= 3:
            paths.append(shell)
        for interior in polygon.interiors:
            hole = _ring_to_path(list(interior.coords)[:-1])
            if len(hole) >= 3:
                paths.append(hole)
    return paths


def _ring_to_path(ring: list[tuple[float, float]] | list[list[float]] | list[Any]) -> list[tuple[int, int]]:
    return [
        (
            int(round(float(x) * CLIPPER_SCALE)),
            int(round(float(y) * CLIPPER_SCALE)),
        )
        for x, y in ring
    ]


def _polytree_to_polygons(tree: Any) -> list[Polygon]:
    polygons: list[Polygon] = []
    for child in getattr(tree, "Childs", []):
        polygons.extend(_collect_polynode(child))
    return _extract_polygons(unary_union(polygons)) if polygons else []


def _collect_polynode(node: Any) -> list[Polygon]:
    polygons: list[Polygon] = []
    if getattr(node, "IsHole", False):
        for child in getattr(node, "Childs", []):
            polygons.extend(_collect_polynode(child))
        return polygons

    shell = _path_to_ring(getattr(node, "Contour", []))
    holes: list[list[tuple[float, float]]] = []
    for child in getattr(node, "Childs", []):
        if getattr(child, "IsHole", False):
            hole = _path_to_ring(getattr(child, "Contour", []))
            if len(hole) >= 3:
                holes.append(hole)
            for grandchild in getattr(child, "Childs", []):
                polygons.extend(_collect_polynode(grandchild))
        else:
            polygons.extend(_collect_polynode(child))

    if len(shell) >= 3:
        polygons.append(Polygon(shell, holes))
    return polygons


def _path_to_ring(path: list[tuple[int, int]] | list[list[int]]) -> list[tuple[float, float]]:
    return [
        (
            float(x) / CLIPPER_SCALE,
            float(y) / CLIPPER_SCALE,
        )
        for x, y in path
    ]


def _extract_polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0 else []
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if polygon.area > 0]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for child in geometry.geoms:
            polygons.extend(_extract_polygons(child))
        return polygons
    return []
