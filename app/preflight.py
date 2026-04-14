from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRIANGLE_EPSILON = 1e-9
RAY_EPSILON = 1e-8
COPLANAR_EPSILON = 1e-6
EXACT_DUPLICATE_VOLUME_RATIO = 0.98
EXACT_DUPLICATE_AABB_RATIO = 0.98
CONTAINMENT_OVERLAP_RATIO = 0.98


@dataclass(slots=True)
class PreflightValidationResult:
    payload: dict[str, Any]


@dataclass(slots=True)
class MeshSpace:
    entity: dict[str, Any]
    vertices: np.ndarray
    faces: np.ndarray
    triangles: np.ndarray
    triangle_aabb_min: np.ndarray
    triangle_aabb_max: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    centroid: np.ndarray
    volume_m3: float


def run_preflight_validation(
    job_id: str,
    job_dir: Path,
    preprocessing_result: dict[str, Any],
    *,
    clash_tolerance_m: float,
) -> PreflightValidationResult:
    geometry_dir = job_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    all_spaces = sorted(
        [entity for entity in preprocessing_result.get("entities", []) if entity.get("entity_type") == "IfcSpace"],
        key=_entity_sort_key,
    )

    checked_spaces: list[MeshSpace] = []
    for entity in all_spaces:
        if entity.get("repair_actions"):
            warnings.append(
                {
                    "code": "auto_repair_applied",
                    "message": "Automatic mesh cleanup actions were applied to the normalized space mesh.",
                    "entity": _entity_ref(entity),
                    "repair_actions": list(entity.get("repair_actions", [])),
                }
            )

        if not _is_valid_space_solid(entity):
            blockers.append(
                {
                    "code": "invalid_space_solid",
                    "message": "Normalized space solid is missing or topologically invalid.",
                    "entity": _entity_ref(entity),
                    "reason": _space_invalid_reason(entity),
                }
            )
            warnings.append(
                {
                    "code": "space_skipped_nonblocking",
                    "message": "Skipped advanced preflight checks because the normalized space solid is invalid.",
                    "entity": _entity_ref(entity),
                    "reason": _space_invalid_reason(entity),
                }
            )
            continue

        checked_spaces.append(_build_mesh_space(entity))

    mesh_spaces_by_express_id = {mesh_space.entity["express_id"]: mesh_space for mesh_space in checked_spaces}
    self_intersections: list[dict[str, Any]] = []
    clash_pair_records: list[dict[str, Any]] = []
    clash_candidates: list[MeshSpace] = []
    for mesh_space in checked_spaces:
        self_intersection = _find_self_intersection(mesh_space)
        if self_intersection is None:
            clash_candidates.append(mesh_space)
            continue

        self_intersections.append(
            {
                "express_id": mesh_space.entity["express_id"],
                "global_id": mesh_space.entity.get("global_id"),
                "name": mesh_space.entity.get("name"),
                "sample_points": {
                    "triangle_a_centroid": _round_vector(self_intersection["triangle_a_centroid"]),
                    "triangle_b_centroid": _round_vector(self_intersection["triangle_b_centroid"]),
                },
            }
        )

    for pair_index, (left_space, right_space) in enumerate(
        _iter_mesh_pair_candidates(clash_candidates, tolerance=clash_tolerance_m)
    ):
        clash = _detect_mesh_clash(left_space, right_space)
        if clash is None:
            continue
        clash_pair_records.append(
            _build_clash_pair_record(
                pair_index,
                left_space,
                right_space,
                clash,
                tolerance=clash_tolerance_m,
            )
        )

    clash_groups = _build_clash_groups(
        pair_records=clash_pair_records,
        self_intersections=self_intersections,
        mesh_spaces_by_express_id=mesh_spaces_by_express_id,
    )
    blockers.extend(_build_clash_group_blockers(clash_groups))

    recommended_resolution = _build_preflight_recommended_resolution(clash_groups)
    review_required = bool(clash_groups)
    resolution_status = "review_required" if review_required else "clear"
    clash_report_payload = {
        "job_id": job_id,
        "status": "failed" if clash_groups else "passed",
        "summary": _build_clash_summary(clash_groups),
        "clash_groups": clash_groups,
        "recommended_resolution": recommended_resolution,
        "resolution_status": resolution_status,
        "review_required": review_required,
    }

    artifacts = {
        "detail": "geometry/preflight.json",
        "clash_report": "geometry/clash_report.json",
    }
    payload = {
        "job_id": job_id,
        "status": "failed" if blockers else "passed",
        "summary": {
            "space_count": len(all_spaces),
            "checked_space_count": len(checked_spaces),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "self_intersection_space_count": len(self_intersections),
            "clash_pair_count": len(clash_pair_records),
            **_build_clash_summary(clash_groups),
        },
        "blockers": blockers,
        "warnings": warnings,
        "clash_groups": clash_groups,
        "recommended_resolution": recommended_resolution,
        "resolution_status": resolution_status,
        "review_required": review_required,
        "artifacts": artifacts,
    }
    _write_json(geometry_dir / "preflight.json", payload)
    _write_json(geometry_dir / "clash_report.json", clash_report_payload)
    return PreflightValidationResult(payload=payload)


def _is_valid_space_solid(entity: dict[str, Any]) -> bool:
    mesh = entity.get("mesh") or {}
    return bool(
        entity.get("valid")
        and mesh.get("vertices")
        and mesh.get("faces")
        and entity.get("closed")
        and entity.get("manifold")
        and float(entity.get("volume_m3") or 0.0) > 0.0
    )


def _space_invalid_reason(entity: dict[str, Any]) -> str:
    reasons: list[str] = []
    mesh = entity.get("mesh") or {}
    if not mesh.get("vertices") or not mesh.get("faces"):
        reasons.append("Mesh data unavailable")
    if not entity.get("closed"):
        reasons.append("Mesh is open")
    if not entity.get("manifold"):
        reasons.append("Mesh is non-manifold")
    if float(entity.get("volume_m3") or 0.0) <= 0.0:
        reasons.append("Non-positive volume")
    if entity.get("reason"):
        reasons.append(str(entity["reason"]))
    return "; ".join(dict.fromkeys(reasons)) or "Space mesh unavailable"


def _build_mesh_space(entity: dict[str, Any]) -> MeshSpace:
    vertices = np.asarray(entity["mesh"]["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(entity["mesh"]["faces"], dtype=np.int64).reshape(-1, 3)
    triangles = vertices[faces]
    triangle_aabb_min = np.min(triangles, axis=1)
    triangle_aabb_max = np.max(triangles, axis=1)
    return MeshSpace(
        entity=entity,
        vertices=vertices,
        faces=faces,
        triangles=triangles,
        triangle_aabb_min=triangle_aabb_min,
        triangle_aabb_max=triangle_aabb_max,
        aabb_min=np.min(vertices, axis=0),
        aabb_max=np.max(vertices, axis=0),
        centroid=np.mean(vertices, axis=0),
        volume_m3=float(entity.get("volume_m3") or 0.0),
    )


def _build_clash_pair_record(
    pair_index: int,
    left_space: MeshSpace,
    right_space: MeshSpace,
    clash: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    classification, evidence = _classify_clash_pair(left_space, right_space, clash, tolerance=tolerance)
    pair_record = {
        "pair_id": f"pair_{pair_index}",
        "entities": [_entity_ref(left_space.entity), _entity_ref(right_space.entity)],
        "pair": {
            "space_a_global_id": left_space.entity.get("global_id"),
            "space_a_express_id": left_space.entity["express_id"],
            "space_b_global_id": right_space.entity.get("global_id"),
            "space_b_express_id": right_space.entity["express_id"],
        },
        "detection": clash["type"],
        "classification": classification,
        "evidence": evidence,
    }
    if clash.get("sample_points"):
        pair_record["sample_points"] = {
            key: _round_vector(value) for key, value in clash["sample_points"].items()
        }
    if clash.get("sample_point") is not None:
        pair_record["sample_point"] = _round_vector(clash["sample_point"])
    return pair_record


def _classify_clash_pair(
    left_space: MeshSpace,
    right_space: MeshSpace,
    clash: dict[str, Any],
    *,
    tolerance: float,
) -> tuple[str, dict[str, Any]]:
    overlap_extents = _aabb_overlap_extents(left_space.aabb_min, left_space.aabb_max, right_space.aabb_min, right_space.aabb_max)
    overlap_volume = float(np.prod(overlap_extents)) if np.all(overlap_extents > TRIANGLE_EPSILON) else 0.0
    smaller_volume = min(left_space.volume_m3, right_space.volume_m3)
    larger_volume = max(left_space.volume_m3, right_space.volume_m3)
    volume_ratio = (smaller_volume / larger_volume) if larger_volume > 0.0 else 0.0
    overlap_to_smaller_volume_ratio = (overlap_volume / smaller_volume) if smaller_volume > 0.0 else 0.0
    same_aabb = bool(
        np.allclose(left_space.aabb_min, right_space.aabb_min, atol=tolerance)
        and np.allclose(left_space.aabb_max, right_space.aabb_max, atol=tolerance)
    )
    left_centroid_in_right = _point_inside_mesh(left_space.centroid, right_space)
    right_centroid_in_left = _point_inside_mesh(right_space.centroid, left_space)

    evidence = {
        "same_aabb": same_aabb,
        "volume_ratio": round(float(volume_ratio), 6),
        "overlap_aabb_volume_m3": round(float(overlap_volume), 6),
        "overlap_to_smaller_volume_ratio": round(float(overlap_to_smaller_volume_ratio), 6),
        "left_centroid_in_right": left_centroid_in_right,
        "right_centroid_in_left": right_centroid_in_left,
    }

    if (
        same_aabb
        and volume_ratio >= EXACT_DUPLICATE_VOLUME_RATIO
        and overlap_to_smaller_volume_ratio >= EXACT_DUPLICATE_AABB_RATIO
        and left_centroid_in_right
        and right_centroid_in_left
    ):
        return "exact_duplicate", evidence

    if overlap_to_smaller_volume_ratio >= CONTAINMENT_OVERLAP_RATIO:
        if left_centroid_in_right and left_space.volume_m3 < right_space.volume_m3:
            evidence["contained_space_express_id"] = left_space.entity["express_id"]
            evidence["keeper_space_express_id"] = right_space.entity["express_id"]
            return "contained_fragment", evidence
        if right_centroid_in_left and right_space.volume_m3 < left_space.volume_m3:
            evidence["contained_space_express_id"] = right_space.entity["express_id"]
            evidence["keeper_space_express_id"] = left_space.entity["express_id"]
            return "contained_fragment", evidence

    if clash["type"] == "containment":
        if left_centroid_in_right and left_space.volume_m3 <= right_space.volume_m3:
            evidence["contained_space_express_id"] = left_space.entity["express_id"]
            evidence["keeper_space_express_id"] = right_space.entity["express_id"]
            return "contained_fragment", evidence
        if right_centroid_in_left and right_space.volume_m3 <= left_space.volume_m3:
            evidence["contained_space_express_id"] = right_space.entity["express_id"]
            evidence["keeper_space_express_id"] = left_space.entity["express_id"]
            return "contained_fragment", evidence

    return "partial_overlap", evidence


def _build_clash_groups(
    *,
    pair_records: list[dict[str, Any]],
    self_intersections: list[dict[str, Any]],
    mesh_spaces_by_express_id: dict[int, MeshSpace],
) -> list[dict[str, Any]]:
    pair_records_by_component: list[list[dict[str, Any]]] = []
    adjacency: dict[int, set[int]] = {}
    pair_ids_by_member: dict[int, list[dict[str, Any]]] = {}
    for pair_record in pair_records:
        left_id = int(pair_record["pair"]["space_a_express_id"])
        right_id = int(pair_record["pair"]["space_b_express_id"])
        adjacency.setdefault(left_id, set()).add(right_id)
        adjacency.setdefault(right_id, set()).add(left_id)
        pair_ids_by_member.setdefault(left_id, []).append(pair_record)
        pair_ids_by_member.setdefault(right_id, []).append(pair_record)

    visited: set[int] = set()
    components: list[list[int]] = []
    for express_id in sorted(adjacency):
        if express_id in visited:
            continue
        stack = [express_id]
        component: list[int] = []
        visited.add(express_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    clash_groups: list[dict[str, Any]] = []
    for component_index, component_ids in enumerate(components):
        component_pairs = [
            pair_record
            for pair_record in pair_records
            if int(pair_record["pair"]["space_a_express_id"]) in component_ids
            and int(pair_record["pair"]["space_b_express_id"]) in component_ids
        ]
        member_spaces = [mesh_spaces_by_express_id[express_id] for express_id in component_ids]
        clash_groups.append(
            _build_pair_clash_group(
                component_index,
                member_spaces,
                component_pairs,
            )
        )

    for self_intersection in sorted(self_intersections, key=lambda entry: entry["express_id"]):
        mesh_space = mesh_spaces_by_express_id.get(int(self_intersection["express_id"]))
        if mesh_space is None:
            continue
        clash_groups.append(
            _build_self_intersection_group(
                len(clash_groups),
                mesh_space,
                self_intersection,
            )
        )

    return sorted(
        clash_groups,
        key=lambda group: (
            min(member["express_id"] for member in group["spaces"]),
            group["clash_group_id"],
        ),
    )


def _build_pair_clash_group(
    component_index: int,
    member_spaces: list[MeshSpace],
    pair_records: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_classifications = {pair_record["classification"] for pair_record in pair_records}
    if "partial_overlap" in pair_classifications:
        classification = "partial_overlap"
    elif "contained_fragment" in pair_classifications:
        classification = "contained_fragment"
    else:
        classification = "exact_duplicate"

    recommended_resolution = None
    if classification == "exact_duplicate":
        recommended_resolution = _build_exact_duplicate_resolution(member_spaces)
    elif classification == "contained_fragment":
        recommended_resolution = _build_contained_fragment_resolution(member_spaces, pair_records)

    resolution_status = "recommended" if recommended_resolution is not None else "manual_required"
    spaces = _serialize_group_spaces(member_spaces, recommended_resolution)
    return {
        "clash_group_id": f"cg_{component_index}",
        "classification": classification,
        "review_required": True,
        "resolution_status": resolution_status,
        "recommended_resolution": recommended_resolution,
        "spaces": spaces,
        "pairs": pair_records,
    }


def _build_self_intersection_group(
    component_index: int,
    mesh_space: MeshSpace,
    self_intersection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "clash_group_id": f"cg_{component_index}",
        "classification": "self_intersection_only",
        "review_required": True,
        "resolution_status": "manual_required",
        "recommended_resolution": None,
        "spaces": _serialize_group_spaces([mesh_space], None),
        "pairs": [],
        "self_intersection": {
            "entity": _entity_ref(mesh_space.entity),
            "sample_points": self_intersection["sample_points"],
        },
    }


def _build_exact_duplicate_resolution(member_spaces: list[MeshSpace]) -> dict[str, Any]:
    keeper = max(member_spaces, key=_keeper_rank_key)
    spaces_to_remove = [mesh_space for mesh_space in member_spaces if mesh_space.entity["express_id"] != keeper.entity["express_id"]]
    return {
        "operation": "remove_spaces",
        "reason": "Keep the highest-ranked duplicate space and remove the overlapping duplicates.",
        "keeper": _entity_ref(keeper.entity),
        "spaces_to_remove": [_entity_ref(mesh_space.entity) for mesh_space in spaces_to_remove],
    }


def _build_contained_fragment_resolution(
    member_spaces: list[MeshSpace],
    pair_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    keeper = max(member_spaces, key=_keeper_rank_key)
    keeper_id = int(keeper.entity["express_id"])
    removable_ids: set[int] = set()
    for pair_record in pair_records:
        if pair_record["classification"] not in {"contained_fragment", "exact_duplicate"}:
            return None
        left_id = int(pair_record["pair"]["space_a_express_id"])
        right_id = int(pair_record["pair"]["space_b_express_id"])
        if pair_record["classification"] == "contained_fragment":
            if int(pair_record["evidence"].get("keeper_space_express_id") or -1) != keeper_id:
                return None
            removable_ids.add(int(pair_record["evidence"]["contained_space_express_id"]))
            continue
        candidate_ids = {left_id, right_id}
        candidate_ids.discard(keeper_id)
        removable_ids.update(candidate_ids)

    if not removable_ids:
        return None
    spaces_to_remove = [
        mesh_space
        for mesh_space in member_spaces
        if mesh_space.entity["express_id"] in removable_ids and mesh_space.entity["express_id"] != keeper_id
    ]
    if not spaces_to_remove:
        return None
    return {
        "operation": "remove_spaces",
        "reason": "Keep the containing space and remove smaller fully-contained fragment spaces.",
        "keeper": _entity_ref(keeper.entity),
        "spaces_to_remove": [_entity_ref(mesh_space.entity) for mesh_space in spaces_to_remove],
    }


def _serialize_group_spaces(
    member_spaces: list[MeshSpace],
    recommended_resolution: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    keeper_express_id = (
        int(recommended_resolution["keeper"]["express_id"])
        if recommended_resolution and recommended_resolution.get("keeper")
        else None
    )
    removable_ids = {
        int(space_ref["express_id"])
        for space_ref in (recommended_resolution or {}).get("spaces_to_remove", [])
    }
    spaces = []
    for mesh_space in sorted(member_spaces, key=lambda candidate: _entity_sort_key(candidate.entity)):
        recommended_action = None
        if keeper_express_id is not None and mesh_space.entity["express_id"] == keeper_express_id:
            recommended_action = "keep"
        elif mesh_space.entity["express_id"] in removable_ids:
            recommended_action = "remove"
        spaces.append(
            {
                **_entity_ref(mesh_space.entity),
                "volume_m3": round(float(mesh_space.volume_m3), 6),
                "recommended_action": recommended_action,
            }
        )
    return spaces


def _build_preflight_recommended_resolution(clash_groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    recommended_groups = [
        clash_group
        for clash_group in clash_groups
        if clash_group.get("recommended_resolution") is not None
    ]
    if not recommended_groups:
        return None

    remove_space_refs: dict[int, dict[str, Any]] = {}
    for clash_group in recommended_groups:
        for space_ref in clash_group["recommended_resolution"]["spaces_to_remove"]:
            remove_space_refs[int(space_ref["express_id"])] = {
                "global_id": space_ref.get("global_id"),
                "express_id": int(space_ref["express_id"]),
            }

    return {
        "operation": "resolve_space_clashes",
        "group_count": len(recommended_groups),
        "clash_group_ids": [clash_group["clash_group_id"] for clash_group in recommended_groups],
        "space_global_ids": [
            space_ref["global_id"]
            for _, space_ref in sorted(remove_space_refs.items())
            if space_ref.get("global_id")
        ],
        "space_express_ids": [space_ref["express_id"] for _, space_ref in sorted(remove_space_refs.items())],
    }


def _build_clash_group_blockers(clash_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for clash_group in clash_groups:
        classification = clash_group["classification"]
        if classification == "self_intersection_only":
            blockers.append(
                {
                    "code": "self_intersection",
                    "message": "Normalized space mesh contains intersecting triangles.",
                    "clash_group_id": clash_group["clash_group_id"],
                    "classification": classification,
                    "review_required": True,
                    "resolution_status": clash_group["resolution_status"],
                    "entity": clash_group["spaces"][0],
                    "self_intersection": clash_group.get("self_intersection"),
                }
            )
            continue

        if classification == "exact_duplicate":
            message = "Detected duplicate intersecting IfcSpace volumes that require review before continuing."
        elif classification == "contained_fragment":
            message = "Detected fully contained IfcSpace fragments that require review before continuing."
        else:
            message = "Detected partially overlapping IfcSpace volumes that must be resolved before continuing."

        blockers.append(
            {
                "code": "space_clash",
                "message": message,
                "clash_group_id": clash_group["clash_group_id"],
                "classification": classification,
                "review_required": True,
                "resolution_status": clash_group["resolution_status"],
                "entities": clash_group["spaces"],
                "pairs": clash_group["pairs"],
                "recommended_resolution": clash_group["recommended_resolution"],
            }
        )
    return blockers


def _build_clash_summary(clash_groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clash_group_count": len(clash_groups),
        "exact_duplicate_group_count": sum(1 for clash_group in clash_groups if clash_group["classification"] == "exact_duplicate"),
        "contained_fragment_group_count": sum(
            1 for clash_group in clash_groups if clash_group["classification"] == "contained_fragment"
        ),
        "partial_overlap_group_count": sum(
            1 for clash_group in clash_groups if clash_group["classification"] == "partial_overlap"
        ),
        "self_intersection_group_count": sum(
            1 for clash_group in clash_groups if clash_group["classification"] == "self_intersection_only"
        ),
        "recommended_resolution_group_count": sum(
            1 for clash_group in clash_groups if clash_group.get("recommended_resolution") is not None
        ),
        "manual_resolution_group_count": sum(
            1 for clash_group in clash_groups if clash_group.get("recommended_resolution") is None
        ),
    }


def _keeper_rank_key(mesh_space: MeshSpace) -> tuple[int, float, int, int]:
    entity = mesh_space.entity
    return (
        int(bool(entity.get("valid") and entity.get("closed") and entity.get("manifold"))),
        float(mesh_space.volume_m3),
        int(bool(entity.get("global_id"))),
        -int(entity["express_id"]),
    )


def _find_self_intersection(mesh_space: MeshSpace) -> dict[str, np.ndarray] | None:
    for left_index, right_index in _iter_self_triangle_candidates(mesh_space):
        if _triangles_share_vertex(mesh_space.faces[left_index], mesh_space.faces[right_index]):
            continue
        left_triangle = mesh_space.triangles[left_index]
        right_triangle = mesh_space.triangles[right_index]
        if not _triangles_intersect_non_coplanar(left_triangle, right_triangle):
            continue
        return {
            "triangle_a_centroid": np.mean(left_triangle, axis=0),
            "triangle_b_centroid": np.mean(right_triangle, axis=0),
        }
    return None


def _detect_mesh_clash(left_space: MeshSpace, right_space: MeshSpace) -> dict[str, Any] | None:
    if not _aabb_has_positive_volume_overlap(left_space.aabb_min, left_space.aabb_max, right_space.aabb_min, right_space.aabb_max):
        return None

    for left_index, right_index in _iter_triangle_pair_candidates(left_space, right_space):
        left_triangle = left_space.triangles[left_index]
        right_triangle = right_space.triangles[right_index]
        if not _triangles_intersect_non_coplanar(left_triangle, right_triangle):
            continue
        return {
            "type": "triangle_intersection",
            "sample_points": {
                "triangle_a_centroid": np.mean(left_triangle, axis=0),
                "triangle_b_centroid": np.mean(right_triangle, axis=0),
            },
        }

    if _point_inside_mesh(left_space.centroid, right_space):
        return {"type": "containment", "sample_point": left_space.centroid}
    if _point_inside_mesh(right_space.centroid, left_space):
        return {"type": "containment", "sample_point": right_space.centroid}
    return None


def _iter_mesh_pair_candidates(
    spaces: list[MeshSpace],
    *,
    tolerance: float,
) -> list[tuple[MeshSpace, MeshSpace]]:
    sorted_spaces = sorted(spaces, key=lambda space: (float(space.aabb_min[0]), *_entity_sort_key(space.entity)))
    active: list[MeshSpace] = []
    pairs: list[tuple[MeshSpace, MeshSpace]] = []

    for current in sorted_spaces:
        min_x = float(current.aabb_min[0])
        active = [space for space in active if float(space.aabb_max[0]) + tolerance >= min_x]
        for candidate in active:
            if _aabb_overlap(candidate.aabb_min, candidate.aabb_max, current.aabb_min, current.aabb_max, tolerance):
                pairs.append((candidate, current))
        active.append(current)
    return pairs


def _iter_self_triangle_candidates(mesh_space: MeshSpace) -> list[tuple[int, int]]:
    sorted_indices = sorted(range(len(mesh_space.triangles)), key=lambda index: float(mesh_space.triangle_aabb_min[index, 0]))
    active: list[int] = []
    pairs: list[tuple[int, int]] = []

    for current_index in sorted_indices:
        current_min_x = float(mesh_space.triangle_aabb_min[current_index, 0])
        active = [
            index for index in active if float(mesh_space.triangle_aabb_max[index, 0]) + TRIANGLE_EPSILON >= current_min_x
        ]
        for candidate_index in active:
            if _aabb_overlap(
                mesh_space.triangle_aabb_min[candidate_index],
                mesh_space.triangle_aabb_max[candidate_index],
                mesh_space.triangle_aabb_min[current_index],
                mesh_space.triangle_aabb_max[current_index],
                TRIANGLE_EPSILON,
            ):
                pairs.append((candidate_index, current_index))
        active.append(current_index)
    return pairs


def _iter_triangle_pair_candidates(left_space: MeshSpace, right_space: MeshSpace) -> list[tuple[int, int]]:
    sorted_left = sorted(range(len(left_space.triangles)), key=lambda index: float(left_space.triangle_aabb_min[index, 0]))
    sorted_right = sorted(range(len(right_space.triangles)), key=lambda index: float(right_space.triangle_aabb_min[index, 0]))

    pairs: list[tuple[int, int]] = []
    right_pointer = 0
    active_right: list[int] = []
    for left_index in sorted_left:
        left_min_x = float(left_space.triangle_aabb_min[left_index, 0])
        left_max_x = float(left_space.triangle_aabb_max[left_index, 0])
        while right_pointer < len(sorted_right):
            right_index = sorted_right[right_pointer]
            if float(right_space.triangle_aabb_min[right_index, 0]) > left_max_x + TRIANGLE_EPSILON:
                break
            active_right.append(right_index)
            right_pointer += 1

        active_right = [
            index for index in active_right if float(right_space.triangle_aabb_max[index, 0]) + TRIANGLE_EPSILON >= left_min_x
        ]
        for right_index in active_right:
            if _aabb_overlap(
                left_space.triangle_aabb_min[left_index],
                left_space.triangle_aabb_max[left_index],
                right_space.triangle_aabb_min[right_index],
                right_space.triangle_aabb_max[right_index],
                TRIANGLE_EPSILON,
            ):
                pairs.append((left_index, right_index))
    return pairs


def _aabb_overlap(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    tolerance: float,
) -> bool:
    return bool(
        np.all(left_min <= right_max + tolerance)
        and np.all(right_min <= left_max + tolerance)
    )


def _aabb_has_positive_volume_overlap(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
) -> bool:
    overlap_extents = _aabb_overlap_extents(left_min, left_max, right_min, right_max)
    return bool(np.all(overlap_extents > TRIANGLE_EPSILON))


def _aabb_overlap_extents(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
) -> np.ndarray:
    return np.minimum(left_max, right_max) - np.maximum(left_min, right_min)


def _triangles_share_vertex(left_face: np.ndarray, right_face: np.ndarray) -> bool:
    return bool(np.any(left_face[:, np.newaxis] == right_face[np.newaxis, :]))


def _triangles_intersect_non_coplanar(left_triangle: np.ndarray, right_triangle: np.ndarray) -> bool:
    left_normal = np.cross(left_triangle[1] - left_triangle[0], left_triangle[2] - left_triangle[0])
    right_normal = np.cross(right_triangle[1] - right_triangle[0], right_triangle[2] - right_triangle[0])
    left_norm = float(np.linalg.norm(left_normal))
    right_norm = float(np.linalg.norm(right_normal))
    if left_norm <= TRIANGLE_EPSILON or right_norm <= TRIANGLE_EPSILON:
        return False

    left_normal /= left_norm
    right_normal /= right_norm
    if abs(float(np.dot(left_normal, right_normal))) >= 1.0 - COPLANAR_EPSILON:
        left_plane_distances = np.dot(right_triangle - left_triangle[0], left_normal)
        right_plane_distances = np.dot(left_triangle - right_triangle[0], right_normal)
        if np.all(np.abs(left_plane_distances) <= TRIANGLE_EPSILON) and np.all(np.abs(right_plane_distances) <= TRIANGLE_EPSILON):
            return False

    left_plane_distances = np.dot(right_triangle - left_triangle[0], left_normal)
    if np.all(left_plane_distances > TRIANGLE_EPSILON) or np.all(left_plane_distances < -TRIANGLE_EPSILON):
        return False

    right_plane_distances = np.dot(left_triangle - right_triangle[0], right_normal)
    if np.all(right_plane_distances > TRIANGLE_EPSILON) or np.all(right_plane_distances < -TRIANGLE_EPSILON):
        return False

    for start, end in _triangle_edges(left_triangle):
        if _segment_intersects_triangle_interior(start, end, right_triangle):
            return True
    for start, end in _triangle_edges(right_triangle):
        if _segment_intersects_triangle_interior(start, end, left_triangle):
            return True
    return False


def _triangle_edges(triangle: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return (
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    )


def _segment_intersects_triangle_interior(start: np.ndarray, end: np.ndarray, triangle: np.ndarray) -> bool:
    direction = end - start
    edge_one = triangle[1] - triangle[0]
    edge_two = triangle[2] - triangle[0]
    cross_direction = np.cross(direction, edge_two)
    determinant = float(np.dot(edge_one, cross_direction))
    if abs(determinant) <= TRIANGLE_EPSILON:
        return False

    inverse_determinant = 1.0 / determinant
    offset = start - triangle[0]
    barycentric_u = inverse_determinant * float(np.dot(offset, cross_direction))
    if barycentric_u < -TRIANGLE_EPSILON or barycentric_u > 1.0 + TRIANGLE_EPSILON:
        return False

    offset_cross = np.cross(offset, edge_one)
    barycentric_v = inverse_determinant * float(np.dot(direction, offset_cross))
    if barycentric_v < -TRIANGLE_EPSILON or barycentric_u + barycentric_v > 1.0 + TRIANGLE_EPSILON:
        return False

    distance = inverse_determinant * float(np.dot(edge_two, offset_cross))
    return bool(-TRIANGLE_EPSILON <= distance <= 1.0 + TRIANGLE_EPSILON)


def _point_inside_mesh(point: np.ndarray, mesh_space: MeshSpace) -> bool:
    if any(_point_on_triangle(point, triangle) for triangle in mesh_space.triangles):
        return False

    directions = (
        np.asarray([1.0, 0.37, 0.53], dtype=np.float64),
        np.asarray([0.23, 1.0, 0.71], dtype=np.float64),
        np.asarray([0.61, 0.17, 1.0], dtype=np.float64),
    )
    votes = 0
    valid_directions = 0
    for direction in directions:
        direction /= np.linalg.norm(direction)
        intersections: list[float] = []
        for triangle in mesh_space.triangles:
            distance = _ray_intersects_triangle(point, direction, triangle)
            if distance is None:
                continue
            intersections.append(distance)

        if not intersections:
            valid_directions += 1
            continue

        intersections.sort()
        unique_intersections: list[float] = []
        for distance in intersections:
            if unique_intersections and abs(unique_intersections[-1] - distance) <= RAY_EPSILON:
                continue
            unique_intersections.append(distance)

        valid_directions += 1
        if len(unique_intersections) % 2 == 1:
            votes += 1
    return valid_directions > 0 and votes >= max(1, (valid_directions // 2) + 1)


def _ray_intersects_triangle(origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray) -> float | None:
    edge_one = triangle[1] - triangle[0]
    edge_two = triangle[2] - triangle[0]
    cross_direction = np.cross(direction, edge_two)
    determinant = float(np.dot(edge_one, cross_direction))
    if abs(determinant) <= TRIANGLE_EPSILON:
        return None

    inverse_determinant = 1.0 / determinant
    offset = origin - triangle[0]
    barycentric_u = inverse_determinant * float(np.dot(offset, cross_direction))
    if barycentric_u <= TRIANGLE_EPSILON or barycentric_u >= 1.0 - TRIANGLE_EPSILON:
        return None

    offset_cross = np.cross(offset, edge_one)
    barycentric_v = inverse_determinant * float(np.dot(direction, offset_cross))
    if barycentric_v <= TRIANGLE_EPSILON or barycentric_u + barycentric_v >= 1.0 - TRIANGLE_EPSILON:
        return None

    distance = inverse_determinant * float(np.dot(edge_two, offset_cross))
    if distance <= RAY_EPSILON:
        return None
    return distance


def _point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    edge_one = triangle[1] - triangle[0]
    edge_two = triangle[2] - triangle[0]
    normal = np.cross(edge_one, edge_two)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= TRIANGLE_EPSILON:
        return False

    unit_normal = normal / normal_norm
    plane_distance = abs(float(np.dot(point - triangle[0], unit_normal)))
    if plane_distance > RAY_EPSILON:
        return False

    vector = point - triangle[0]
    dot00 = float(np.dot(edge_two, edge_two))
    dot01 = float(np.dot(edge_two, edge_one))
    dot02 = float(np.dot(edge_two, vector))
    dot11 = float(np.dot(edge_one, edge_one))
    dot12 = float(np.dot(edge_one, vector))
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) <= TRIANGLE_EPSILON:
        return False

    inverse_denominator = 1.0 / denominator
    barycentric_u = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
    barycentric_v = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
    return bool(
        -RAY_EPSILON <= barycentric_u <= 1.0 + RAY_EPSILON
        and -RAY_EPSILON <= barycentric_v <= 1.0 + RAY_EPSILON
        and barycentric_u + barycentric_v <= 1.0 + RAY_EPSILON
    )


def _entity_sort_key(entity: dict[str, Any]) -> tuple[str, int]:
    return (str(entity.get("global_id") or ""), int(entity["express_id"]))


def _entity_ref(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_id": entity.get("global_id"),
        "express_id": entity["express_id"],
        "name": entity.get("name"),
    }


def _round_vector(vector: np.ndarray | list[float]) -> list[float]:
    return [round(float(value), 6) for value in vector]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
