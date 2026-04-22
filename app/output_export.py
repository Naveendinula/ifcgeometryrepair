from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .gbxml_export import build_gbxml_preflight_payload, export_gbxml_from_preflight_payload
from .mesh_normalizer import build_obj_text


# ---------------------------------------------------------------------------
# Minimum-area filtering (Paper Section 9.2)
# ---------------------------------------------------------------------------

DEFAULT_MIN_SURFACE_AREA_M2 = 0.25


def _filter_surfaces_by_area(
    surfaces: list[dict[str, Any]],
    min_area_m2: float,
    area_key: str = "area_m2",
) -> list[dict[str, Any]]:
    """Remove surfaces whose area falls below *min_area_m2* (Paper Sec 9.2).

    Surfaces with small areas contribute minimally to the BEM while increasing
    simulation time.  The paper uses A_min = 0.25 m² as a practical default.
    """
    return [s for s in surfaces if float(s.get(area_key, 0.0)) >= min_area_m2]


# ---------------------------------------------------------------------------
# Opening-subtracted boundary helpers (Paper Section 3.3.4)
# ---------------------------------------------------------------------------


def _get_modified_boundaries(
    opening_integration_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a lookup of boundary surfaces modified by opening subtraction.

    Per the paper (Section 3.3.4), after opening surfaces are generated the
    previously computed 2LSB-2a surfaces must be updated by subtracting the
    opening areas.  ``opening_integration`` already computes these as
    ``modified_boundaries`` with ``remainder_polygon_rings_3d`` and
    ``remainder_area_m2``.
    """
    modified: dict[str, dict[str, Any]] = {}
    for boundary in opening_integration_payload.get("modified_boundaries", []):
        surface_id = boundary.get("surface_id", "")
        if surface_id and boundary.get("opening_surface_ids"):
            modified[surface_id] = boundary
    return modified


def _apply_opening_subtraction(
    surface: dict[str, Any],
    modified_boundaries: dict[str, dict[str, Any]],
    surface_id_key: str = "surface_id",
) -> dict[str, Any]:
    """Return a copy of *surface* with opening areas subtracted if applicable.

    When a boundary was modified during opening integration, its
    ``polygon_rings_3d`` and ``area_m2`` are replaced with the remainder
    geometry (i.e. the boundary minus the opening footprints).
    """
    sid = surface.get(surface_id_key, "")
    modified = modified_boundaries.get(sid)
    if modified is None:
        return surface
    updated = dict(surface)
    remainder_rings = modified.get("remainder_polygon_rings_3d")
    remainder_area = modified.get("remainder_area_m2")
    if remainder_rings is not None:
        updated["polygon_rings_3d"] = remainder_rings
    if remainder_area is not None:
        updated["area_m2"] = remainder_area
    updated["opening_surface_ids"] = modified.get("opening_surface_ids", [])
    return updated


# ---------------------------------------------------------------------------
# XML export
# ---------------------------------------------------------------------------


def export_2lsb_xml(
    opening_integration_payload: dict[str, Any],
    internal_boundary_payload: dict[str, Any],
    external_shell_payload: dict[str, Any],
    output_path: Path,
    *,
    min_area_m2: float = DEFAULT_MIN_SURFACE_AREA_M2,
) -> Path:
    """Serialize the final 2LSB surface set to XML.

    Combines internal boundaries, classified external surfaces, and opening
    integration results into a single ``<SecondLevelSpaceBoundaries>`` document.

    Per the paper (Section 3.3.4), boundary surfaces are exported with opening
    areas subtracted (using ``modified_boundaries`` from opening integration).
    Surfaces below *min_area_m2* are excluded (Paper Section 9.2).
    """
    modified_boundaries = _get_modified_boundaries(opening_integration_payload)

    root = ET.Element("SecondLevelSpaceBoundaries")

    # --- Internal boundary surfaces (2LSB-2a) ---
    internal_el = ET.SubElement(root, "InternalBoundaries")
    internal_surfaces = _filter_surfaces_by_area(
        internal_boundary_payload.get("oriented_surfaces", []),
        min_area_m2,
    )
    for surface in internal_surfaces:
        updated = _apply_opening_subtraction(
            surface, modified_boundaries, surface_id_key="oriented_surface_id",
        )
        _add_surface_element(
            internal_el,
            surface_id=updated.get("oriented_surface_id", ""),
            classification="internal",
            surface=updated,
        )

    # --- External surfaces (2LSB-2a external / 2LSB-2b) ---
    external_el = ET.SubElement(root, "ExternalBoundaries")
    external_surfaces = _filter_surfaces_by_area(
        external_shell_payload.get("surfaces", []),
        min_area_m2,
    )
    for surface in external_surfaces:
        updated = _apply_opening_subtraction(surface, modified_boundaries)
        _add_surface_element(
            external_el,
            surface_id=updated.get("surface_id", ""),
            classification=updated.get("classification", "unclassified"),
            surface=updated,
        )

    # --- Opening surfaces ---
    openings_el = ET.SubElement(root, "OpeningSurfaces")
    opening_surfaces = _filter_surfaces_by_area(
        opening_integration_payload.get("opening_surfaces", []),
        min_area_m2,
    )
    for surface in opening_surfaces:
        _add_surface_element(
            openings_el,
            surface_id=surface.get("surface_id", ""),
            classification="opening",
            surface=surface,
        )

    # --- Summary ---
    summary_el = ET.SubElement(root, "Summary")
    oi_summary = opening_integration_payload.get("summary", {})
    ib_summary = internal_boundary_payload.get("summary", {})
    es_summary = external_shell_payload.get("summary", {})
    ET.SubElement(summary_el, "InternalSurfaceCount").text = str(len(internal_surfaces))
    ET.SubElement(summary_el, "ExternalSurfaceCount").text = str(len(external_surfaces))
    ET.SubElement(summary_el, "OpeningSurfaceCount").text = str(len(opening_surfaces))
    ET.SubElement(summary_el, "TotalOpeningAreaM2").text = str(
        round(sum(float(s.get("area_m2", 0.0)) for s in opening_surfaces), 6)
    )
    ET.SubElement(summary_el, "MinSurfaceAreaThresholdM2").text = str(
        round(min_area_m2, 6)
    )

    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def _add_surface_element(
    parent: ET.Element,
    *,
    surface_id: str,
    classification: str,
    surface: dict[str, Any],
) -> ET.Element:
    el = ET.SubElement(parent, "Surface", id=surface_id)
    ET.SubElement(el, "Classification").text = classification

    space_gid = surface.get("space_global_id")
    space_eid = surface.get("space_express_id")
    if space_gid or space_eid is not None:
        space_el = ET.SubElement(el, "Space")
        if space_gid:
            space_el.set("globalId", str(space_gid))
        if space_eid is not None:
            space_el.set("expressId", str(space_eid))
        if surface.get("space_name"):
            space_el.set("name", surface["space_name"])

    adjacent_gid = surface.get("adjacent_space_global_id")
    adjacent_eid = surface.get("adjacent_space_express_id")
    if adjacent_gid or adjacent_eid is not None:
        adj_el = ET.SubElement(el, "AdjacentSpace")
        if adjacent_gid:
            adj_el.set("globalId", str(adjacent_gid))
        if adjacent_eid is not None:
            adj_el.set("expressId", str(adjacent_eid))

    opening_eid = surface.get("opening_express_id")
    if opening_eid is not None:
        opening_el = ET.SubElement(el, "Opening")
        opening_el.set("expressId", str(opening_eid))
        if surface.get("opening_global_id"):
            opening_el.set("globalId", surface["opening_global_id"])

    normal = surface.get("normal") or surface.get("plane_normal")
    if normal:
        ET.SubElement(el, "Normal").text = _format_vector(normal)

    centroid = surface.get("centroid")
    if centroid:
        ET.SubElement(el, "Centroid").text = _format_vector(centroid)

    area = surface.get("area_m2")
    if area is not None:
        ET.SubElement(el, "AreaM2").text = str(round(float(area), 6))

    rings = surface.get("polygon_rings_3d")
    if rings:
        geom_el = ET.SubElement(el, "Geometry")
        for ring_index, ring in enumerate(rings):
            ring_el = ET.SubElement(geom_el, "Ring", index=str(ring_index))
            for point in ring:
                ET.SubElement(ring_el, "Point").text = _format_vector(point)

    return el


def _format_vector(v: list[float] | tuple[float, ...]) -> str:
    return " ".join(str(round(float(x), 6)) for x in v)


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

CLASSIFICATION_ORDER = [
    "internal",
    "external_wall",
    "roof",
    "ground_floor",
    "opening",
    "internal_void",
    "unclassified",
]


def export_2lsb_obj(
    opening_integration_payload: dict[str, Any],
    internal_boundary_payload: dict[str, Any],
    external_shell_payload: dict[str, Any],
    output_path: Path,
    *,
    min_area_m2: float = DEFAULT_MIN_SURFACE_AREA_M2,
) -> Path:
    """Export all 2LSB surfaces as a single OBJ file with named groups.

    Boundaries are exported with opening areas subtracted (Paper Section 3.3.4).
    Surfaces below *min_area_m2* are excluded (Paper Section 9.2).
    """
    modified_boundaries = _get_modified_boundaries(opening_integration_payload)
    meshes: list[dict[str, Any]] = []

    # Internal boundary oriented surfaces.
    internal_surfaces = _filter_surfaces_by_area(
        internal_boundary_payload.get("oriented_surfaces", []),
        min_area_m2,
    )
    for surface in internal_surfaces:
        triangles = surface.get("triangles", [])
        mesh = _triangles_to_flat_mesh(triangles)
        if mesh:
            mesh["name"] = f"internal_{surface.get('oriented_surface_id', '')}"
            meshes.append(mesh)

    # External classified surfaces.
    external_surfaces = _filter_surfaces_by_area(
        external_shell_payload.get("surfaces", []),
        min_area_m2,
    )
    for surface in external_surfaces:
        triangles = surface.get("triangles", [])
        if triangles:
            mesh = _triangles_to_flat_mesh(triangles)
            if mesh:
                classification = surface.get("classification", "unclassified")
                mesh["name"] = f"{classification}_{surface.get('surface_id', '')}"
                meshes.append(mesh)

    # Opening surfaces from integration.
    opening_surfaces = _filter_surfaces_by_area(
        opening_integration_payload.get("opening_surfaces", []),
        min_area_m2,
    )
    for surface in opening_surfaces:
        triangles = surface.get("triangles", [])
        mesh = _triangles_to_flat_mesh(triangles)
        if mesh:
            mesh["name"] = f"opening_{surface.get('surface_id', '')}"
            meshes.append(mesh)

    # Sort by classification order for readability.
    def _sort_key(m: dict[str, Any]) -> tuple[int, str]:
        name = m.get("name", "")
        for index, cls in enumerate(CLASSIFICATION_ORDER):
            if name.startswith(cls):
                return (index, name)
        return (len(CLASSIFICATION_ORDER), name)

    meshes.sort(key=_sort_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_obj_text(meshes), encoding="utf-8")
    return output_path


def _triangles_to_flat_mesh(
    triangles: list[list[list[float]]],
) -> dict[str, Any] | None:
    """Convert a list of 3D triangle vertex lists into a flat vertices/faces mesh."""
    if not triangles:
        return None
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for tri in triangles:
        if len(tri) != 3:
            continue
        base = len(vertices)
        for point in tri:
            vertices.append([float(v) for v in point])
        faces.append([base, base + 1, base + 2])
    if not faces:
        return None
    return {"vertices": vertices, "faces": faces}


# ---------------------------------------------------------------------------
# gbXML export  (Paper Section 3.4)
# ---------------------------------------------------------------------------

GBXML_NS = "http://www.gbxml.org/schema"


def export_2lsb_gbxml(
    opening_integration_payload: dict[str, Any],
    internal_boundary_payload: dict[str, Any],
    external_shell_payload: dict[str, Any],
    output_path: Path,
    *,
    min_area_m2: float = 0.0,
    gbxml_tolerance_m: float = 1e-3,
    preprocessing_result: dict[str, Any] | None = None,
    gbxml_preflight_payload: dict[str, Any] | None = None,
) -> Path:
    """Export a gbXML document using the gbXML-specific preflight/export plan."""
    payload = gbxml_preflight_payload or build_gbxml_preflight_payload(
        preprocessing_result,
        internal_boundary_payload,
        external_shell_payload,
        opening_integration_payload,
        tolerance_m=gbxml_tolerance_m,
        min_area_m2=min_area_m2,
    )
    return export_gbxml_from_preflight_payload(payload, output_path)


def _collect_space_ids(
    surfaces: list[dict[str, Any]],
    space_ids: set[str],
) -> None:
    for surface in surfaces:
        for key in ("space_global_id", "space_a_global_id", "space_b_global_id"):
            gid = surface.get(key)
            if gid:
                space_ids.add(str(gid))


def _sanitize_id(raw_id: str) -> str:
    """Produce an XML-safe ID string."""
    return raw_id.replace(" ", "_").replace("/", "_") if raw_id else "unknown"


def _classification_to_gbxml_type(classification: str) -> str:
    mapping = {
        "external_wall": "ExteriorWall",
        "roof": "Roof",
        "ground_floor": "UndergroundSlab",
        "internal_void": "InteriorWall",
        "internal": "InteriorWall",
        "opening": "Air",
    }
    return mapping.get(classification, "ExteriorWall")


def _add_gbxml_surface(
    parent: ET.Element,
    *,
    surface_id: str,
    surface_type: str,
    surface: dict[str, Any],
    space_a_key: str = "space_global_id",
    space_b_key: str | None = None,
) -> ET.Element:
    el = ET.SubElement(parent, "Surface", id=_sanitize_id(surface_id))
    el.set("surfaceType", surface_type)

    # Adjacent space references.
    space_a = surface.get(space_a_key)
    if space_a:
        adj_a = ET.SubElement(el, "AdjacentSpaceId")
        adj_a.set("spaceIdRef", _sanitize_id(str(space_a)))

    space_b = surface.get(space_b_key) if space_b_key else surface.get("adjacent_space_global_id")
    if space_b:
        adj_b = ET.SubElement(el, "AdjacentSpaceId")
        adj_b.set("spaceIdRef", _sanitize_id(str(space_b)))

    # Planar geometry.
    rings = surface.get("polygon_rings_3d")
    if rings and rings[0]:
        pg_el = ET.SubElement(el, "PlanarGeometry")
        polyloop = ET.SubElement(pg_el, "PolyLoop")
        for point in rings[0]:
            cp = ET.SubElement(polyloop, "CartesianPoint")
            for coord_value in point:
                ET.SubElement(cp, "Coordinate").text = str(round(float(coord_value), 6))

    return el


def _add_gbxml_opening(
    surface_el: ET.Element,
    opening: dict[str, Any],
) -> ET.Element:
    """Add an ``<Opening>`` sub-element to a gbXML ``<Surface>``."""
    opening_type = "OperableWindow"  # Default; could be refined from IFC data.
    boundary_cls = opening.get("boundary_classification", "")
    if boundary_cls == "internal":
        opening_type = "NonSlidingDoor"

    oid = _sanitize_id(opening.get("surface_id", ""))
    el = ET.SubElement(surface_el, "Opening", id=oid)
    el.set("openingType", opening_type)

    rings = opening.get("polygon_rings_3d")
    if rings and rings[0]:
        pg_el = ET.SubElement(el, "PlanarGeometry")
        polyloop = ET.SubElement(pg_el, "PolyLoop")
        for point in rings[0]:
            cp = ET.SubElement(polyloop, "CartesianPoint")
            for coord_value in point:
                ET.SubElement(cp, "Coordinate").text = str(round(float(coord_value), 6))

    return el
