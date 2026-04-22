from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient as orient_polygon
from shapely.ops import unary_union


GBXML_NS = "http://www.gbxml.org/schema"
AREA_EPSILON_M2 = 1e-9


@dataclass(slots=True)
class GbxmlPreflightResult:
    payload: dict[str, Any]


def run_gbxml_preflight(
    job_id: str,
    job_dir: Path,
    preprocessing_result: dict[str, Any],
    internal_boundary_result: dict[str, Any],
    external_shell_result: dict[str, Any],
    opening_integration_result: dict[str, Any],
    *,
    tolerance_m: float,
    min_area_m2: float = 0.0,
) -> GbxmlPreflightResult:
    payload = build_gbxml_preflight_payload(
        preprocessing_result,
        internal_boundary_result,
        external_shell_result,
        opening_integration_result,
        tolerance_m=tolerance_m,
        min_area_m2=min_area_m2,
    )
    payload["job_id"] = job_id
    payload["artifacts"] = {"detail": "geometry/gbxml_preflight.json"}
    _write_json(job_dir / "geometry" / "gbxml_preflight.json", payload)
    return GbxmlPreflightResult(payload=payload)


def export_gbxml_from_preflight_payload(
    gbxml_preflight_payload: dict[str, Any],
    output_path: Path,
) -> Path:
    root = ET.Element("gbXML")
    root.set("xmlns", GBXML_NS)
    root.set("temperatureUnit", "C")
    root.set("lengthUnit", "Meters")
    root.set("areaUnit", "SquareMeters")
    root.set("volumeUnit", "CubicMeters")
    root.set("version", "6.01")

    campus_el = ET.SubElement(root, "Campus", id="campus_1")
    building_el = ET.SubElement(campus_el, "Building", id="building_1")
    building_el.set("buildingType", "Unknown")

    export_plan = gbxml_preflight_payload.get("export_plan", {})
    for space in export_plan.get("spaces", []):
        space_id = _sanitize_id(str(space.get("id") or "unknown"))
        space_el = ET.SubElement(building_el, "Space", id=space_id)
        ET.SubElement(space_el, "Name").text = str(space.get("name") or space.get("id") or "Unnamed Space")

    for surface in export_plan.get("surfaces", []):
        surface_el = ET.SubElement(campus_el, "Surface", id=surface.get("xml_id") or _sanitize_id(surface["id"]))
        surface_el.set("surfaceType", str(surface.get("surface_type") or "ExteriorWall"))
        for adjacent_space_id in surface.get("adjacent_space_ids", []):
            adj_el = ET.SubElement(surface_el, "AdjacentSpaceId")
            adj_el.set("spaceIdRef", _sanitize_id(str(adjacent_space_id)))

        planar_geometry = ET.SubElement(surface_el, "PlanarGeometry")
        polyloop = ET.SubElement(planar_geometry, "PolyLoop")
        for point in surface.get("rings_3d", [[]])[0]:
            cartesian_point = ET.SubElement(polyloop, "CartesianPoint")
            for coordinate in point:
                ET.SubElement(cartesian_point, "Coordinate").text = str(round(float(coordinate), 6))

        for opening in surface.get("openings", []):
            opening_el = ET.SubElement(surface_el, "Opening", id=opening.get("xml_id") or _sanitize_id(opening["id"]))
            opening_el.set("openingType", str(opening.get("opening_type") or "OperableWindow"))
            opening_geometry = ET.SubElement(opening_el, "PlanarGeometry")
            opening_loop = ET.SubElement(opening_geometry, "PolyLoop")
            for point in opening.get("rings_3d", [[]])[0]:
                cartesian_point = ET.SubElement(opening_loop, "CartesianPoint")
                for coordinate in point:
                    ET.SubElement(cartesian_point, "Coordinate").text = str(round(float(coordinate), 6))

    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def build_gbxml_preflight_payload(
    preprocessing_result: dict[str, Any] | None,
    internal_boundary_result: dict[str, Any],
    external_shell_result: dict[str, Any],
    opening_integration_result: dict[str, Any],
    *,
    tolerance_m: float = 1e-3,
    min_area_m2: float = 0.0,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    omitted_entities: list[dict[str, Any]] = []

    space_index = _index_spaces(preprocessing_result or {})
    space_stats = {
        space_id: {
            "space_id": space_id,
            "name": space_data.get("name"),
            "valid_geometry": bool(space_data.get("valid_geometry", True)),
            "selected_surface_count": 0,
            "exported_surface_count": 0,
            "omitted_surface_count": 0,
            "blocker_count": 0,
            "warning_count": 0,
        }
        for space_id, space_data in space_index.items()
    }

    _append_internal_midpoint_rejections(
        internal_boundary_result=internal_boundary_result,
        blockers=blockers,
        omitted_entities=omitted_entities,
    )

    oriented_to_shared = {
        surface.get("oriented_surface_id"): surface.get("shared_surface_id")
        for surface in internal_boundary_result.get("oriented_surfaces", [])
        if surface.get("oriented_surface_id") and surface.get("shared_surface_id")
    }
    external_geometry_by_id = _index_external_surface_geometry(external_shell_result, opening_integration_result)
    export_surfaces: list[dict[str, Any]] = []

    for surface in internal_boundary_result.get("shared_surfaces", []):
        source_surface_id = str(surface.get("shared_surface_id") or "").strip()
        if not source_surface_id:
            continue
        adjacent_space_ids = [
            _space_ref(surface.get("space_a_global_id"), surface.get("space_a_express_id")),
            _space_ref(surface.get("space_b_global_id"), surface.get("space_b_express_id")),
        ]
        export_surfaces.extend(
            _prepare_parent_surface(
                surface=surface,
                source_surface_id=source_surface_id,
                source_kind="internal_shared",
                surface_type="InteriorWall",
                adjacent_space_ids=[space_id for space_id in adjacent_space_ids if space_id],
                space_index=space_index,
                space_stats=space_stats,
                warnings=warnings,
                blockers=blockers,
                omitted_entities=omitted_entities,
                tolerance_m=tolerance_m,
                min_area_m2=min_area_m2,
            )
        )

    _append_internal_midpoint_missing_adjacencies(
        internal_boundary_result=internal_boundary_result,
        blockers=blockers,
        omitted_entities=omitted_entities,
    )

    _GBXML_EXCLUDED_CLASSIFICATIONS = frozenset({"unclassified"})
    excluded_surface_ids: set[str] = set()

    for serialized_surface in external_shell_result.get("surfaces", []):
        source_surface_id = str(serialized_surface.get("surface_id") or "").strip()
        if not source_surface_id:
            continue
        shell_classification = serialized_surface.get("classification", "unclassified")
        if shell_classification in _GBXML_EXCLUDED_CLASSIFICATIONS:
            excluded_surface_ids.add(source_surface_id)
            omitted_entities.append(
                _omitted_ref("surface", source_surface_id, f"shell_classification_excluded:{shell_classification}")
            )
            continue
        geometry_surface = external_geometry_by_id.get(source_surface_id) or serialized_surface
        adjacent_space_ids = [
            _space_ref(geometry_surface.get("space_global_id"), geometry_surface.get("space_express_id")),
        ]
        export_surfaces.extend(
            _prepare_parent_surface(
                surface=geometry_surface,
                source_surface_id=source_surface_id,
                source_kind="external",
                surface_type=_classification_to_gbxml_type(shell_classification),
                adjacent_space_ids=[space_id for space_id in adjacent_space_ids if space_id],
                space_index=space_index,
                space_stats=space_stats,
                warnings=warnings,
                blockers=blockers,
                omitted_entities=omitted_entities,
                tolerance_m=tolerance_m,
                min_area_m2=min_area_m2,
            )
        )

    surface_by_source_id: dict[str, list[dict[str, Any]]] = {}
    for surface in export_surfaces:
        surface_by_source_id.setdefault(surface["source_surface_id"], []).append(surface)

    export_openings: list[dict[str, Any]] = []
    for opening in opening_integration_result.get("opening_surfaces", []):
        parent_source_id = str(opening.get("boundary_surface_id") or "").strip()
        if parent_source_id in oriented_to_shared:
            parent_source_id = str(oriented_to_shared[parent_source_id])

        # If the parent surface was excluded by classification filter, omit the
        # opening with a warning (not a blocker) — the parent is intentionally
        # absent from the export, so the child opening cannot appear either.
        if parent_source_id in excluded_surface_ids:
            opening_id = str(opening.get("surface_id") or "unknown_opening")
            warnings.append(
                _issue(
                    "opening_parent_excluded",
                    f'Opening "{opening_id}" omitted because its parent surface '
                    f'"{parent_source_id}" was excluded by classification filter.',
                    entity=_opening_ref(opening),
                )
            )
            omitted_entities.append(_omitted_ref("opening", opening_id, "opening_parent_excluded"))
            continue

        export_openings.extend(
            _prepare_openings(
                opening=opening,
                parent_source_id=parent_source_id,
                parent_components=surface_by_source_id.get(parent_source_id, []),
                warnings=warnings,
                blockers=blockers,
                omitted_entities=omitted_entities,
                tolerance_m=tolerance_m,
                min_area_m2=min_area_m2,
            )
        )

    for opening in export_openings:
        for surface in surface_by_source_id.get(opening["parent_source_surface_id"], []):
            if surface["id"] == opening["parent_surface_id"]:
                surface.setdefault("openings", []).append(opening)
                break

    spaces_to_export: set[str] = set()
    for surface in export_surfaces:
        for space_id in surface.get("adjacent_space_ids", []):
            spaces_to_export.add(space_id)
            stats = space_stats.setdefault(
                space_id,
                {
                    "space_id": space_id,
                    "name": None,
                    "valid_geometry": True,
                    "selected_surface_count": 0,
                    "exported_surface_count": 0,
                    "omitted_surface_count": 0,
                    "blocker_count": 0,
                    "warning_count": 0,
                },
            )
            stats["exported_surface_count"] += 1

    _apply_unique_ids(export_surfaces, "surface", blockers)
    _apply_unique_ids(export_openings, "opening", blockers)
    _apply_unique_ids(
        [{"id": space_id, "entity": {"space_id": space_id}} for space_id in spaces_to_export],
        "space",
        blockers,
    )

    warning_space_ids = _space_ids_from_issues(warnings)
    blocker_space_ids = _space_ids_from_issues(blockers)
    zone_summary = []
    zone_status_counts = {"complete": 0, "partial": 0, "invalid": 0}
    for space_id in sorted(spaces_to_export):
        stats = space_stats.get(space_id) or {}
        warning_count = int(stats.get("warning_count", 0)) + warning_space_ids.count(space_id)
        blocker_count = int(stats.get("blocker_count", 0)) + blocker_space_ids.count(space_id)
        exported_surface_count = int(stats.get("exported_surface_count", 0))
        omitted_surface_count = int(stats.get("omitted_surface_count", 0))
        if not stats.get("valid_geometry", True) or exported_surface_count == 0 or blocker_count > 0 and omitted_surface_count >= exported_surface_count:
            zone_status = "invalid"
        elif blocker_count > 0 or omitted_surface_count > 0 or warning_count > 0:
            zone_status = "partial"
        else:
            zone_status = "complete"
        zone_status_counts[zone_status] += 1
        zone_summary.append(
            {
                "space_id": space_id,
                "name": stats.get("name"),
                "status": zone_status,
                "exported_surface_count": exported_surface_count,
                "omitted_surface_count": omitted_surface_count,
                "warning_count": warning_count,
                "blocker_count": blocker_count,
            }
        )

    status = "valid"
    if blockers:
        status = "invalid"
    elif warnings:
        status = "warning"

    return {
        "status": status,
        "tolerance_m": round(float(tolerance_m), 6),
        "min_area_m2": round(float(min_area_m2), 6),
        "summary": {
            "space_count": len(spaces_to_export),
            "surface_count": len(export_surfaces),
            "opening_count": len(export_openings),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "omitted_entity_count": len(omitted_entities),
            "zone_status_counts": zone_status_counts,
        },
        "warnings": warnings,
        "blockers": blockers,
        "omitted_entities": omitted_entities,
        "zone_summary": zone_summary,
        "artifacts": {},
        "export_plan": {
            "spaces": [
                {
                    "id": space_id,
                    "name": (space_stats.get(space_id) or {}).get("name") or space_id,
                }
                for space_id in sorted(spaces_to_export)
            ],
            "surfaces": _serialize_export_surfaces(export_surfaces),
        },
    }


def _append_internal_midpoint_rejections(
    *,
    internal_boundary_result: dict[str, Any],
    blockers: list[dict[str, Any]],
    omitted_entities: list[dict[str, Any]],
) -> None:
    for rejection in internal_boundary_result.get("rejected_shared_components", []):
        shared_surface_id = str(rejection.get("shared_surface_id") or "unknown_internal_midpoint")
        issue_code = _internal_midpoint_issue_code(str(rejection.get("rejection_code") or ""))
        blockers.append(
            _issue(
                issue_code,
                str(rejection.get("rejection_message") or f'Internal midpoint surface "{shared_surface_id}" was rejected.'),
                entity=_internal_midpoint_entity(rejection),
            )
        )
        omitted_entities.append(
            _omitted_ref(
                "surface",
                shared_surface_id,
                issue_code,
                space_ids=_internal_midpoint_space_ids(rejection),
            )
        )


def _append_internal_midpoint_missing_adjacencies(
    *,
    internal_boundary_result: dict[str, Any],
    blockers: list[dict[str, Any]],
    omitted_entities: list[dict[str, Any]],
) -> None:
    rejected_keys = {
        _internal_adjacency_key_from_refs(rejection.get("space_a_global_id"), rejection.get("space_a_express_id"))
        + "::"
        + _internal_adjacency_key_from_refs(rejection.get("space_b_global_id"), rejection.get("space_b_express_id"))
        for rejection in internal_boundary_result.get("rejected_shared_components", [])
    }
    for adjacency in internal_boundary_result.get("adjacencies", []):
        if not adjacency.get("oriented_surface_ids") or adjacency.get("shared_surface_ids"):
            continue
        adjacency_key = (
            _internal_adjacency_key_from_refs(adjacency.get("space_a_global_id"), adjacency.get("space_a_express_id"))
            + "::"
            + _internal_adjacency_key_from_refs(adjacency.get("space_b_global_id"), adjacency.get("space_b_express_id"))
        )
        if adjacency_key in rejected_keys:
            continue
        space_ids = [
            _space_ref(adjacency.get("space_a_global_id"), adjacency.get("space_a_express_id")),
            _space_ref(adjacency.get("space_b_global_id"), adjacency.get("space_b_express_id")),
        ]
        adjacency_id = (
            f'internal_adjacency_{_sanitize_id(str(space_ids[0] or adjacency.get("space_a_express_id") or "a"))}'
            f'__{_sanitize_id(str(space_ids[1] or adjacency.get("space_b_express_id") or "b"))}'
        )
        blockers.append(
            _issue(
                "internal_midpoint_surface_missing",
                "Internal adjacency produced oriented overlaps but no shared midpoint surface for gbXML export.",
                entity={
                    "surface_id": adjacency_id,
                    "space_ids": [space_id for space_id in space_ids if space_id],
                    "oriented_surface_ids": list(adjacency.get("oriented_surface_ids", [])),
                },
            )
        )
        omitted_entities.append(
            _omitted_ref(
                "surface",
                adjacency_id,
                "internal_midpoint_surface_missing",
                space_ids=[space_id for space_id in space_ids if space_id],
            )
        )


def _prepare_parent_surface(
    *,
    surface: dict[str, Any],
    source_surface_id: str,
    source_kind: str,
    surface_type: str,
    adjacent_space_ids: list[str],
    space_index: dict[str, dict[str, Any]],
    space_stats: dict[str, dict[str, Any]],
    warnings: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    omitted_entities: list[dict[str, Any]],
    tolerance_m: float,
    min_area_m2: float,
) -> list[dict[str, Any]]:
    if not adjacent_space_ids:
        blockers.append(
            _issue(
                "missing_adjacent_space",
                f'Surface "{source_surface_id}" has no adjacent space reference for gbXML export.',
                entity=_surface_ref(surface, source_surface_id),
            )
        )
        omitted_entities.append(_omitted_ref("surface", source_surface_id, "missing_adjacent_space"))
        return []

    if source_kind == "internal_shared" and len(adjacent_space_ids) < 2:
        blockers.append(
            _issue(
                "internal_surface_missing_adjacent_space",
                f'Internal shared surface "{source_surface_id}" is missing one or both adjacent spaces.',
                entity=_surface_ref(surface, source_surface_id),
            )
        )
        omitted_entities.append(_omitted_ref("surface", source_surface_id, "internal_surface_missing_adjacent_space"))
        return []

    for space_id in adjacent_space_ids:
        stats = space_stats.setdefault(
            space_id,
            {
                "space_id": space_id,
                "name": None,
                "valid_geometry": True,
                "selected_surface_count": 0,
                "exported_surface_count": 0,
                "omitted_surface_count": 0,
                "blocker_count": 0,
                "warning_count": 0,
            },
        )
        stats["selected_surface_count"] += 1
        if not space_index.get(space_id, {}).get("valid_geometry", True):
            blockers.append(
                _issue(
                    "invalid_export_space",
                    f'Space "{space_id}" is not closed or has non-positive volume, so surface "{source_surface_id}" cannot be exported safely.',
                    entity={"space_id": space_id, "surface_id": source_surface_id},
                )
            )
            stats["blocker_count"] += 1
            stats["omitted_surface_count"] += 1
            omitted_entities.append(_omitted_ref("surface", source_surface_id, "invalid_export_space", space_id=space_id))
            return []

    components, geometry_warnings = _extract_surface_components(surface, tolerance_m=tolerance_m)
    for warning_code in geometry_warnings:
        warnings.append(
            _issue(
                warning_code,
                f'Surface "{source_surface_id}" required gbXML orientation normalization.',
                entity=_surface_ref(surface, source_surface_id),
            )
        )
        for space_id in adjacent_space_ids:
            space_stats[space_id]["warning_count"] += 1

    if not components:
        blockers.append(
            _issue(
                "surface_unusable_outer_loop",
                f'Surface "{source_surface_id}" does not contain a usable outer loop for gbXML export.',
                entity=_surface_ref(surface, source_surface_id),
            )
        )
        for space_id in adjacent_space_ids:
            space_stats[space_id]["blocker_count"] += 1
            space_stats[space_id]["omitted_surface_count"] += 1
        omitted_entities.append(_omitted_ref("surface", source_surface_id, "surface_unusable_outer_loop"))
        return []

    if len(components) > 1:
        warnings.append(
            _issue(
                "surface_split_into_components",
                f'Surface "{source_surface_id}" was split into {len(components)} gbXML surfaces.',
                entity=_surface_ref(surface, source_surface_id),
            )
        )
        for space_id in adjacent_space_ids:
            space_stats[space_id]["warning_count"] += 1

    prepared_components: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        area_m2 = float(component["polygon_2d"].area)
        if area_m2 <= AREA_EPSILON_M2 or area_m2 < float(min_area_m2):
            warnings.append(
                _issue(
                    "tiny_fragment_pruned",
                    f'Surface "{source_surface_id}" contained a fragment below the gbXML export threshold and it was omitted.',
                    entity=_surface_ref(surface, source_surface_id),
                )
            )
            for space_id in adjacent_space_ids:
                space_stats[space_id]["warning_count"] += 1
                space_stats[space_id]["omitted_surface_count"] += 1
            omitted_entities.append(_omitted_ref("surface", source_surface_id, "tiny_fragment_pruned"))
            continue

        # Prune degenerate sliver fragments: reject polygons whose shortest
        # edge is below a minimum length threshold (nearly-collinear vertices).
        if _is_degenerate_polygon(component["polygon_2d"], min_edge_length=1e-3):
            warnings.append(
                _issue(
                    "degenerate_fragment_pruned",
                    f'Surface "{source_surface_id}" contained a degenerate sliver fragment and it was omitted.',
                    entity=_surface_ref(surface, source_surface_id),
                )
            )
            for space_id in adjacent_space_ids:
                space_stats[space_id]["warning_count"] += 1
                space_stats[space_id]["omitted_surface_count"] += 1
            omitted_entities.append(_omitted_ref("surface", source_surface_id, "degenerate_fragment_pruned"))
            continue

        surface_id = source_surface_id if len(components) == 1 else f"{source_surface_id}__part_{component_index + 1}"
        prepared_components.append(
            {
                "id": surface_id,
                "source_surface_id": source_surface_id,
                "surface_type": surface_type,
                "adjacent_space_ids": list(adjacent_space_ids),
                "rings_3d": component["rings_3d"],
                "polygon_2d": component["polygon_2d"],
                "plane_point": component["plane_point"],
                "basis_u": component["basis_u"],
                "basis_v": component["basis_v"],
                "normal": component["normal"],
                "area_m2": round(area_m2, 6),
                "openings": [],
            }
        )

    if not prepared_components:
        omitted_entities.append(_omitted_ref("surface", source_surface_id, "all_components_omitted"))
    return prepared_components


def _prepare_openings(
    *,
    opening: dict[str, Any],
    parent_source_id: str,
    parent_components: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    omitted_entities: list[dict[str, Any]],
    tolerance_m: float,
    min_area_m2: float,
) -> list[dict[str, Any]]:
    opening_id = str(opening.get("surface_id") or "unknown_opening")
    if not parent_source_id or not parent_components:
        blockers.append(
            _issue(
                "opening_missing_parent_surface",
                f'Opening "{opening_id}" does not resolve to a gbXML parent surface.',
                entity=_opening_ref(opening),
            )
        )
        omitted_entities.append(_omitted_ref("opening", opening_id, "opening_missing_parent_surface"))
        return []

    components, geometry_warnings = _extract_surface_components(opening, tolerance_m=tolerance_m)
    for warning_code in geometry_warnings:
        warnings.append(
            _issue(
                warning_code,
                f'Opening "{opening_id}" required gbXML orientation normalization.',
                entity=_opening_ref(opening),
            )
        )

    if not components:
        blockers.append(
            _issue(
                "surface_unusable_outer_loop",
                f'Opening "{opening_id}" does not contain a usable outer loop for gbXML export.',
                entity=_opening_ref(opening),
            )
        )
        omitted_entities.append(_omitted_ref("opening", opening_id, "surface_unusable_outer_loop"))
        return []

    prepared_openings: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        opening_area_m2 = float(component["polygon_2d"].area)
        if opening_area_m2 <= AREA_EPSILON_M2 or opening_area_m2 < float(min_area_m2):
            warnings.append(
                _issue(
                    "tiny_fragment_pruned",
                    f'Opening "{opening_id}" contained a fragment below the gbXML export threshold and it was omitted.',
                    entity=_opening_ref(opening),
                )
            )
            omitted_entities.append(_omitted_ref("opening", opening_id, "tiny_fragment_pruned"))
            continue

        coplanar_parents: list[dict[str, Any]] = []
        matching_parents: list[dict[str, Any]] = []
        for parent_surface in parent_components:
            if not _is_coplanar(component, parent_surface, tolerance_m=tolerance_m):
                continue
            coplanar_parents.append(parent_surface)
            if parent_surface["polygon_2d"].buffer(float(tolerance_m)).covers(component["polygon_2d"]):
                matching_parents.append(parent_surface)

        if not coplanar_parents:
            blockers.append(
                _issue(
                    "opening_not_coplanar_with_parent",
                    f'Opening "{opening_id}" is not coplanar with its gbXML parent surface "{parent_source_id}" within tolerance.',
                    entity=_opening_ref(opening),
                )
            )
            omitted_entities.append(_omitted_ref("opening", opening_id, "opening_not_coplanar_with_parent"))
            continue

        if not matching_parents:
            blockers.append(
                _issue(
                    "opening_not_bounded_by_parent",
                    f'Opening "{opening_id}" is not fully bounded by its gbXML parent surface "{parent_source_id}".',
                    entity=_opening_ref(opening),
                )
            )
            omitted_entities.append(_omitted_ref("opening", opening_id, "opening_not_bounded_by_parent"))
            continue

        parent_surface = sorted(matching_parents, key=lambda item: float(item["polygon_2d"].area))[0]
        resolved_opening_id = opening_id if len(components) == 1 else f"{opening_id}__part_{component_index + 1}"
        prepared_openings.append(
            {
                "id": resolved_opening_id,
                "parent_source_surface_id": parent_source_id,
                "parent_surface_id": parent_surface["id"],
                "rings_3d": component["rings_3d"],
                "opening_type": _opening_type(opening),
                "area_m2": round(opening_area_m2, 6),
            }
        )

    return prepared_openings


def _extract_surface_components(
    surface: dict[str, Any],
    *,
    tolerance_m: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    del tolerance_m
    normal = _normalized_vector(surface.get("plane_normal") or surface.get("normal") or [0.0, 0.0, 1.0])
    if normal is None:
        return [], []
    plane_point = np.asarray(surface.get("plane_point") or surface.get("centroid") or [0.0, 0.0, 0.0], dtype=np.float64)
    basis_u, basis_v = _plane_basis(normal)

    components_input = surface.get("polygon_components_3d")
    geometry_components: list[dict[str, Any]] = []
    warnings: list[str] = []

    if components_input:
        for component in components_input:
            polygon, corrected = _polygon_from_component_rings(component, plane_point, basis_u, basis_v)
            if polygon is None:
                continue
            if corrected:
                warnings.append("winding_corrected")
            for extracted in _extract_polygons(polygon):
                oriented = orient_polygon(extracted, sign=1.0)
                geometry_components.append(
                    {
                        "polygon_2d": oriented,
                        "rings_3d": _polygon_to_rings_3d(oriented, plane_point, basis_u, basis_v),
                        "plane_point": plane_point,
                        "basis_u": basis_u,
                        "basis_v": basis_v,
                        "normal": normal.tolist(),
                    }
                )

    if not geometry_components and surface.get("polygon_rings_3d"):
        polygon, corrected = _polygon_from_component_rings(surface.get("polygon_rings_3d", []), plane_point, basis_u, basis_v)
        if polygon is not None:
            if corrected:
                warnings.append("winding_corrected")
            for extracted in _extract_polygons(polygon):
                oriented = orient_polygon(extracted, sign=1.0)
                geometry_components.append(
                    {
                        "polygon_2d": oriented,
                        "rings_3d": _polygon_to_rings_3d(oriented, plane_point, basis_u, basis_v),
                        "plane_point": plane_point,
                        "basis_u": basis_u,
                        "basis_v": basis_v,
                        "normal": normal.tolist(),
                    }
                )

    if not geometry_components and surface.get("triangles"):
        union_polygon = _polygon_from_triangles(surface.get("triangles", []), plane_point, basis_u, basis_v)
        for extracted in _extract_polygons(union_polygon):
            oriented = orient_polygon(extracted, sign=1.0)
            geometry_components.append(
                {
                    "polygon_2d": oriented,
                    "rings_3d": _polygon_to_rings_3d(oriented, plane_point, basis_u, basis_v),
                    "plane_point": plane_point,
                    "basis_u": basis_u,
                    "basis_v": basis_v,
                    "normal": normal.tolist(),
                }
            )

    return geometry_components, list(dict.fromkeys(warnings))


def _polygon_from_component_rings(
    rings_3d: list[list[list[float]]],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[Polygon | MultiPolygon | None, bool]:
    if not rings_3d:
        return None, False
    shell = _project_ring_to_basis(rings_3d[0], plane_point, basis_u, basis_v)
    holes = [_project_ring_to_basis(ring, plane_point, basis_u, basis_v) for ring in rings_3d[1:]]
    corrected = _signed_area(shell) < 0.0 or any(_signed_area(hole) > 0.0 for hole in holes)
    polygon = Polygon(shell, holes)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None, corrected
    if isinstance(polygon, (Polygon, MultiPolygon)):
        return polygon, corrected
    extracted = _extract_polygons(polygon)
    if not extracted:
        return None, corrected
    return unary_union(extracted), corrected


def _polygon_from_triangles(
    triangles_3d: list[list[list[float]]],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> Polygon | MultiPolygon | None:
    polygons: list[Polygon] = []
    for triangle in triangles_3d:
        coordinates = _project_ring_to_basis(triangle, plane_point, basis_u, basis_v)
        if len(coordinates) < 3:
            continue
        triangle_polygon = Polygon(coordinates)
        if triangle_polygon.is_empty or triangle_polygon.area <= AREA_EPSILON_M2:
            continue
        polygons.append(triangle_polygon)
    if not polygons:
        return None
    union_polygon = unary_union(polygons)
    if union_polygon.is_empty:
        return None
    if isinstance(union_polygon, (Polygon, MultiPolygon)):
        return union_polygon
    extracted = _extract_polygons(union_polygon)
    return unary_union(extracted) if extracted else None


def _index_spaces(preprocessing_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entity in preprocessing_result.get("entities", []):
        if entity.get("entity_type") != "IfcSpace":
            continue
        space_id = _space_ref(entity.get("global_id"), entity.get("express_id"))
        if not space_id:
            continue
        indexed[space_id] = {
            "space_id": space_id,
            "name": entity.get("name"),
            "valid_geometry": bool(
                entity.get("valid")
                and entity.get("closed")
                and entity.get("manifold")
                and float(entity.get("volume_m3") or 0.0) > 0.0
            ),
        }
    return indexed


def _index_external_surface_geometry(
    external_shell_result: dict[str, Any],
    opening_integration_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for surface in external_shell_result.get("surfaces", []):
        surface_id = surface.get("surface_id")
        if surface_id:
            indexed[str(surface_id)] = surface
    for boundary in opening_integration_result.get("modified_boundaries", []):
        if boundary.get("boundary_type") != "external":
            continue
        surface_id = boundary.get("surface_id")
        if surface_id and surface_id not in indexed:
            indexed[str(surface_id)] = boundary
    return indexed


def _apply_unique_ids(entries: list[dict[str, Any]], entry_type: str, blockers: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for entry in entries:
        raw_id = str(entry.get("id") or "unknown")
        normalized = _sanitize_id(raw_id)
        duplicate_index = seen.get(normalized, 0)
        if duplicate_index:
            blockers.append(
                _issue(
                    "duplicate_id_after_normalization",
                    f'{entry_type.title()} id "{raw_id}" collided after XML id normalization.',
                    entity=entry.get("entity") or {"id": raw_id},
                )
            )
            normalized = f"{normalized}__dup_{duplicate_index + 1}"
        seen[_sanitize_id(raw_id)] = duplicate_index + 1
        entry["xml_id"] = normalized


def _serialize_export_surfaces(export_surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for surface in export_surfaces:
        serialized.append(
            {
                "id": surface.get("id"),
                "xml_id": surface.get("xml_id"),
                "source_surface_id": surface.get("source_surface_id"),
                "surface_type": surface.get("surface_type"),
                "adjacent_space_ids": list(surface.get("adjacent_space_ids", [])),
                "rings_3d": surface.get("rings_3d", []),
                "area_m2": surface.get("area_m2"),
                "openings": [
                    {
                        "id": opening.get("id"),
                        "xml_id": opening.get("xml_id"),
                        "parent_source_surface_id": opening.get("parent_source_surface_id"),
                        "parent_surface_id": opening.get("parent_surface_id"),
                        "rings_3d": opening.get("rings_3d", []),
                        "opening_type": opening.get("opening_type"),
                        "area_m2": opening.get("area_m2"),
                    }
                    for opening in surface.get("openings", [])
                ],
            }
        )
    return serialized


def _space_ids_from_issues(issues: list[dict[str, Any]]) -> list[str]:
    space_ids: list[str] = []
    for issue in issues:
        entity = issue.get("entity") or {}
        listed_space_ids = [space_id for space_id in entity.get("space_ids", []) if space_id]
        if listed_space_ids:
            for space_id in listed_space_ids:
                space_ids.append(str(space_id))
            continue
        if entity.get("space_id"):
            space_ids.append(str(entity["space_id"]))
    return space_ids


def _classification_to_gbxml_type(classification: str) -> str:
    mapping = {
        "external_wall": "ExteriorWall",
        "roof": "Roof",
        "ground_floor": "UndergroundSlab",
        "internal_void": "InteriorWall",
        "opening": "Air",
    }
    return mapping.get(classification, "ExteriorWall")


def _opening_type(opening: dict[str, Any]) -> str:
    if opening.get("boundary_classification") == "internal":
        return "NonSlidingDoor"
    return "OperableWindow"


def _normalized_vector(value: list[float] | tuple[float, ...]) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        return None
    return vector / magnitude


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = np.cross(helper, normal)
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)
    return basis_u, basis_v


def _project_ring_to_basis(
    ring_3d: list[list[float]],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[tuple[float, float]]:
    projected: list[tuple[float, float]] = []
    for point in ring_3d:
        point_array = np.asarray(point, dtype=np.float64)
        relative = point_array - plane_point
        projected.append((float(relative @ basis_u), float(relative @ basis_v)))
    return projected


def _polygon_to_rings_3d(
    polygon: Polygon,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[list[list[float]]]:
    rings = [
        [_round_vector(_lift_point(coordinate, plane_point, basis_u, basis_v)) for coordinate in list(polygon.exterior.coords)[:-1]]
    ]
    for interior in polygon.interiors:
        rings.append(
            [_round_vector(_lift_point(coordinate, plane_point, basis_u, basis_v)) for coordinate in list(interior.coords)[:-1]]
        )
    return rings


def _lift_point(
    coordinate_2d: tuple[float, float] | list[float],
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> list[float]:
    x, y = coordinate_2d
    lifted = plane_point + (float(x) * basis_u) + (float(y) * basis_v)
    return [float(value) for value in lifted.tolist()]


def _extract_polygons(geometry: Polygon | MultiPolygon | GeometryCollection | None) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for child in geometry.geoms:
            polygons.extend(_extract_polygons(child))
        return polygons


def _is_degenerate_polygon(polygon: Polygon, *, min_edge_length: float = 1e-3) -> bool:
    """Return True if the polygon is a sliver with any edge shorter than *min_edge_length*."""
    coords = list(polygon.exterior.coords)
    if len(coords) < 4:  # closed ring needs at least 4 (3 unique + closing)
        return True
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        if (dx * dx + dy * dy) < min_edge_length * min_edge_length:
            return True
    return False
    return []


def _signed_area(ring_2d: list[tuple[float, float]]) -> float:
    if len(ring_2d) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(ring_2d):
        x2, y2 = ring_2d[(index + 1) % len(ring_2d)]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def _is_coplanar(opening_component: dict[str, Any], parent_surface: dict[str, Any], *, tolerance_m: float) -> bool:
    opening_normal = _normalized_vector(opening_component.get("normal", [0.0, 0.0, 1.0]))
    parent_normal = _normalized_vector(parent_surface.get("normal", [0.0, 0.0, 1.0]))
    if opening_normal is None or parent_normal is None:
        return False
    if 1.0 - abs(float(np.dot(opening_normal, parent_normal))) > float(tolerance_m):
        return False
    opening_plane_point = np.asarray(opening_component.get("plane_point", [0.0, 0.0, 0.0]), dtype=np.float64)
    parent_plane_point = np.asarray(parent_surface.get("plane_point", [0.0, 0.0, 0.0]), dtype=np.float64)
    plane_distance = abs(float(parent_normal @ (opening_plane_point - parent_plane_point)))
    return plane_distance <= float(tolerance_m)


def _space_ref(global_id: str | None, express_id: int | None) -> str | None:
    if global_id:
        return str(global_id)
    if express_id is not None:
        return f"space_{express_id}"
    return None


def _surface_ref(surface: dict[str, Any], surface_id: str) -> dict[str, Any]:
    return {
        "surface_id": surface_id,
        "space_id": _space_ref(surface.get("space_global_id"), surface.get("space_express_id")),
    }


def _opening_ref(opening: dict[str, Any]) -> dict[str, Any]:
    return {
        "surface_id": opening.get("surface_id"),
        "space_id": _space_ref(opening.get("space_global_id"), opening.get("space_express_id")),
        "boundary_surface_id": opening.get("boundary_surface_id"),
    }


def _internal_midpoint_issue_code(rejection_code: str) -> str:
    if rejection_code == "component_count_mismatch":
        return "internal_midpoint_component_count_mismatch"
    if rejection_code in {"degenerate_midpoint_overlap", "degenerate_projected_overlap"}:
        return "internal_midpoint_surface_degenerate"
    return "internal_midpoint_surface_rejected"


def _internal_midpoint_space_ids(rejection: dict[str, Any]) -> list[str]:
    return [
        space_id
        for space_id in (
            _space_ref(rejection.get("space_a_global_id"), rejection.get("space_a_express_id")),
            _space_ref(rejection.get("space_b_global_id"), rejection.get("space_b_express_id")),
        )
        if space_id
    ]


def _internal_midpoint_entity(rejection: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "surface_id": rejection.get("shared_surface_id"),
        "space_ids": _internal_midpoint_space_ids(rejection),
        "source_surface_ids": [
            source_id
            for source_id in (rejection.get("source_surface_a_id"), rejection.get("source_surface_b_id"))
            if source_id
        ],
        "source_polygon_ids": [
            polygon_id
            for polygon_id in (rejection.get("source_polygon_a_id"), rejection.get("source_polygon_b_id"))
            if polygon_id
        ],
        "rejection_code": rejection.get("rejection_code"),
        "oriented_surface_ids": list(rejection.get("oriented_surface_ids", [])),
    }
    if payload["space_ids"]:
        payload["space_id"] = payload["space_ids"][0]
    return payload


def _internal_adjacency_key_from_refs(global_id: str | None, express_id: int | None) -> str:
    return str(_space_ref(global_id, express_id) or express_id or "unknown")


def _omitted_ref(
    entity_type: str,
    entity_id: str,
    reason: str,
    *,
    space_id: str | None = None,
    space_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "entity_type": entity_type,
        "id": entity_id,
        "reason": reason,
    }
    if space_id:
        payload["space_id"] = space_id
    normalized_space_ids = [item for item in (space_ids or []) if item]
    if normalized_space_ids:
        payload["space_ids"] = normalized_space_ids
    return payload


def _issue(code: str, message: str, *, entity: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    if entity is not None:
        payload["entity"] = entity
    return payload


def _sanitize_id(raw_id: str) -> str:
    sanitized = str(raw_id or "unknown").replace(" ", "_").replace("/", "_")
    return sanitized or "unknown"


def _round_vector(values: list[float] | np.ndarray) -> list[float]:
    iterable = values.tolist() if isinstance(values, np.ndarray) else values
    return [round(float(value), 6) for value in iterable]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
