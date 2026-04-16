from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .mesh_normalizer import build_obj_text


# ---------------------------------------------------------------------------
# XML export
# ---------------------------------------------------------------------------


def export_2lsb_xml(
    opening_integration_payload: dict[str, Any],
    internal_boundary_payload: dict[str, Any],
    external_shell_payload: dict[str, Any],
    output_path: Path,
) -> Path:
    """Serialize the final 2LSB surface set to XML.

    Combines internal boundaries, classified external surfaces, and opening
    integration results into a single ``<SecondLevelSpaceBoundaries>`` document.
    """
    root = ET.Element("SecondLevelSpaceBoundaries")

    # --- Internal boundary surfaces (2LSB-2a) ---
    internal_el = ET.SubElement(root, "InternalBoundaries")
    for surface in internal_boundary_payload.get("oriented_surfaces", []):
        _add_surface_element(
            internal_el,
            surface_id=surface.get("oriented_surface_id", ""),
            classification="internal",
            surface=surface,
        )

    # --- External surfaces (2LSB-2b) ---
    external_el = ET.SubElement(root, "ExternalBoundaries")
    for surface in external_shell_payload.get("surfaces", []):
        _add_surface_element(
            external_el,
            surface_id=surface.get("surface_id", ""),
            classification=surface.get("classification", "unclassified"),
            surface=surface,
        )

    # --- Opening surfaces ---
    openings_el = ET.SubElement(root, "OpeningSurfaces")
    for surface in opening_integration_payload.get("opening_surfaces", []):
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
    ET.SubElement(summary_el, "InternalSurfaceCount").text = str(
        ib_summary.get("oriented_surface_count", 0)
    )
    ET.SubElement(summary_el, "ExternalSurfaceCount").text = str(
        es_summary.get("candidate_surface_count", 0)
    )
    ET.SubElement(summary_el, "OpeningSurfaceCount").text = str(
        oi_summary.get("opening_surfaces_created", 0)
    )
    ET.SubElement(summary_el, "TotalOpeningAreaM2").text = str(
        round(oi_summary.get("total_opening_area_m2", 0.0), 6)
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
) -> Path:
    """Export all 2LSB surfaces as a single OBJ file with named groups."""
    meshes: list[dict[str, Any]] = []

    # Internal boundary oriented surfaces.
    for surface in internal_boundary_payload.get("oriented_surfaces", []):
        triangles = surface.get("triangles", [])
        mesh = _triangles_to_flat_mesh(triangles)
        if mesh:
            mesh["name"] = f"internal_{surface.get('oriented_surface_id', '')}"
            meshes.append(mesh)

    # External classified surfaces.
    for surface in external_shell_payload.get("surfaces", []):
        # External shell surfaces may not have inline triangles; skip if absent.
        artifacts = surface.get("artifacts", {})
        # We don't have inline mesh for external_shell surfaces in the payload,
        # but the surface_id links to OBJ files.  For the aggregate OBJ we rely
        # on the triangles if present.
        triangles = surface.get("triangles", [])
        if triangles:
            mesh = _triangles_to_flat_mesh(triangles)
            if mesh:
                classification = surface.get("classification", "unclassified")
                mesh["name"] = f"{classification}_{surface.get('surface_id', '')}"
                meshes.append(mesh)

    # Opening surfaces from integration.
    for surface in opening_integration_payload.get("opening_surfaces", []):
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
