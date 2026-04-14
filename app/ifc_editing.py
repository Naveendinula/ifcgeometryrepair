from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ifcopenshell
from ifcopenshell.api.root import remove_product

from .ifc_extractor import extract_entity_record


class InvalidSpaceRemovalRequestError(ValueError):
    pass


class InvalidSpaceResolutionRequestError(ValueError):
    pass


@dataclass(slots=True)
class SpaceRemovalEditResult:
    requested_space_global_ids: list[str]
    requested_space_express_ids: list[int]
    removed_spaces: list[dict[str, Any]]
    remaining_space_count: int


@dataclass(slots=True)
class SpaceClashResolutionEditResult:
    requested_group_resolutions: list[dict[str, Any]]
    requested_space_global_ids: list[str]
    requested_space_express_ids: list[int]
    removed_spaces: list[dict[str, Any]]
    remaining_space_count: int
    resolved_clash_group_count: int


def derive_ifc_without_spaces(
    input_path: Path,
    output_path: Path,
    *,
    space_global_ids: list[str] | None = None,
    space_express_ids: list[int] | None = None,
) -> SpaceRemovalEditResult:
    requested_space_global_ids = _normalize_global_ids(space_global_ids or [])
    requested_space_express_ids = _normalize_express_ids(space_express_ids or [])
    if not requested_space_global_ids and not requested_space_express_ids:
        raise InvalidSpaceRemovalRequestError("Select at least one IfcSpace to remove.")

    model = ifcopenshell.open(str(input_path))
    all_spaces = list(model.by_type("IfcSpace"))
    if not all_spaces:
        raise InvalidSpaceRemovalRequestError("The IFC file does not contain any IfcSpace entities.")

    resolved_spaces: dict[int, Any] = {}
    unknown_global_ids: list[str] = []
    invalid_global_ids: list[str] = []
    unknown_express_ids: list[int] = []
    invalid_express_ids: list[int] = []

    for global_id in requested_space_global_ids:
        try:
            entity = model.by_guid(global_id)
        except RuntimeError:
            entity = None
        if entity is None:
            unknown_global_ids.append(global_id)
            continue
        if not entity.is_a("IfcSpace"):
            invalid_global_ids.append(global_id)
            continue
        resolved_spaces[entity.id()] = entity

    for express_id in requested_space_express_ids:
        try:
            entity = model.by_id(express_id)
        except RuntimeError:
            entity = None
        if entity is None:
            unknown_express_ids.append(express_id)
            continue
        if not entity.is_a("IfcSpace"):
            invalid_express_ids.append(express_id)
            continue
        resolved_spaces[entity.id()] = entity

    validation_errors: list[str] = []
    if unknown_global_ids:
        validation_errors.append(
            "Unknown space_global_ids: " + ", ".join(sorted(unknown_global_ids))
        )
    if invalid_global_ids:
        validation_errors.append(
            "Non-IfcSpace space_global_ids: " + ", ".join(sorted(invalid_global_ids))
        )
    if unknown_express_ids:
        validation_errors.append(
            "Unknown space_express_ids: " + ", ".join(str(value) for value in sorted(unknown_express_ids))
        )
    if invalid_express_ids:
        validation_errors.append(
            "Non-IfcSpace space_express_ids: " + ", ".join(str(value) for value in sorted(invalid_express_ids))
        )
    if validation_errors:
        raise InvalidSpaceRemovalRequestError(" ".join(validation_errors))

    if not resolved_spaces:
        raise InvalidSpaceRemovalRequestError("No matching IfcSpace entities were found to remove.")

    if len(resolved_spaces) >= len(all_spaces):
        raise InvalidSpaceRemovalRequestError("Cannot remove all remaining IfcSpace entities.")

    removed_spaces = [_removed_space_ref(entity) for entity in resolved_spaces.values()]
    for entity in sorted(resolved_spaces.values(), key=lambda item: item.id(), reverse=True):
        remove_product(model, product=entity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(output_path))

    return SpaceRemovalEditResult(
        requested_space_global_ids=requested_space_global_ids,
        requested_space_express_ids=requested_space_express_ids,
        removed_spaces=sorted(removed_spaces, key=lambda item: item["express_id"]),
        remaining_space_count=len(all_spaces) - len(resolved_spaces),
    )


def derive_ifc_resolving_space_clashes(
    input_path: Path,
    output_path: Path,
    *,
    group_resolutions: list[dict[str, Any]] | None = None,
) -> SpaceClashResolutionEditResult:
    normalized_group_resolutions = _normalize_group_resolutions(group_resolutions or [])
    if not normalized_group_resolutions:
        raise InvalidSpaceResolutionRequestError("Select at least one clash group resolution before creating a rerun.")

    requested_space_global_ids: list[str] = []
    requested_space_express_ids: list[int] = []
    for resolution in normalized_group_resolutions:
        requested_space_global_ids.extend(_normalize_global_ids(resolution.get("remove_space_global_ids", [])))
        requested_space_express_ids.extend(_normalize_express_ids(resolution.get("remove_space_express_ids", [])))

    if not requested_space_global_ids and not requested_space_express_ids:
        raise InvalidSpaceResolutionRequestError("Select at least one IfcSpace to remove from the reviewed clash groups.")

    removal_result = derive_ifc_without_spaces(
        input_path,
        output_path,
        space_global_ids=requested_space_global_ids,
        space_express_ids=requested_space_express_ids,
    )
    return SpaceClashResolutionEditResult(
        requested_group_resolutions=normalized_group_resolutions,
        requested_space_global_ids=removal_result.requested_space_global_ids,
        requested_space_express_ids=removal_result.requested_space_express_ids,
        removed_spaces=removal_result.removed_spaces,
        remaining_space_count=removal_result.remaining_space_count,
        resolved_clash_group_count=len(normalized_group_resolutions),
    )


def _removed_space_ref(entity: Any) -> dict[str, Any]:
    record = extract_entity_record(entity)
    return {
        "global_id": record.get("global_id"),
        "express_id": record["express_id"],
        "name": record.get("name"),
        "entity_type": record["entity_type"],
        "storey": record.get("storey"),
    }


def _normalize_global_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _normalize_express_ids(values: list[int]) -> list[int]:
    seen: set[int] = set()
    normalized: list[int] = []
    for value in values:
        candidate = int(value)
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _normalize_group_resolutions(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for value in values:
        clash_group_id = str(value.get("clash_group_id") or "").strip()
        if not clash_group_id or clash_group_id in seen_group_ids:
            continue
        seen_group_ids.add(clash_group_id)
        normalized.append(
            {
                "clash_group_id": clash_group_id,
                "remove_space_global_ids": _normalize_global_ids(list(value.get("remove_space_global_ids", []))),
                "remove_space_express_ids": _normalize_express_ids(list(value.get("remove_space_express_ids", []))),
            }
        )
    return normalized
