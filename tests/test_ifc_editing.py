from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import ifcopenshell

from app.ifc_editing import derive_ifc_resolving_space_clashes, derive_ifc_without_spaces
from tests.ifc_factory import build_duplicate_rooms_fixture, build_shared_wall_fixture


def test_derive_ifc_without_spaces_removes_space_and_writes_output() -> None:
    fixture = build_shared_wall_fixture()
    root = Path("jobs") / "test_runs" / f"ifc-edit-{uuid4()}"
    root.mkdir(parents=True, exist_ok=False)
    input_path = root / "input.ifc"
    output_path = root / "derived.ifc"

    try:
        input_path.write_bytes(fixture.content)
        original_model = ifcopenshell.open(str(input_path))
        removed_space = original_model.by_type("IfcSpace")[0]

        result = derive_ifc_without_spaces(
            input_path,
            output_path,
            space_global_ids=[removed_space.GlobalId],
        )

        assert output_path.exists()
        assert result.remaining_space_count == 1
        assert result.removed_spaces[0]["global_id"] == removed_space.GlobalId

        derived_model = ifcopenshell.open(str(output_path))
        remaining_space_ids = {space.GlobalId for space in derived_model.by_type("IfcSpace")}
        assert removed_space.GlobalId not in remaining_space_ids
        assert len(remaining_space_ids) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_derive_ifc_resolving_space_clashes_removes_reviewed_spaces() -> None:
    fixture = build_duplicate_rooms_fixture()
    root = Path("jobs") / "test_runs" / f"ifc-clash-edit-{uuid4()}"
    root.mkdir(parents=True, exist_ok=False)
    input_path = root / "input.ifc"
    output_path = root / "derived.ifc"

    try:
        input_path.write_bytes(fixture.content)
        original_model = ifcopenshell.open(str(input_path))
        duplicate_space = original_model.by_type("IfcSpace")[1]

        result = derive_ifc_resolving_space_clashes(
            input_path,
            output_path,
            group_resolutions=[
                {
                    "clash_group_id": "cg_0",
                    "remove_space_global_ids": [duplicate_space.GlobalId],
                    "remove_space_express_ids": [],
                }
            ],
        )

        assert output_path.exists()
        assert result.resolved_clash_group_count == 1
        assert result.removed_spaces[0]["global_id"] == duplicate_space.GlobalId

        derived_model = ifcopenshell.open(str(output_path))
        remaining_space_ids = {space.GlobalId for space in derived_model.by_type("IfcSpace")}
        assert duplicate_space.GlobalId not in remaining_space_ids
        assert len(remaining_space_ids) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
