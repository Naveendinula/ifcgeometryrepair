from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ifcopenshell.api import run


@dataclass(frozen=True, slots=True)
class GeneratedFixture:
    content: bytes
    expected_space_count: int
    expected_opening_count: int
    represented_space_name: str | None = None
    missing_space_name: str | None = None
    opening_name: str | None = None
    storey_name: str | None = None
    building_name: str | None = None
    space_names: tuple[str, ...] = ()
    expected_adjacency_pairs: tuple[tuple[str, str], ...] = ()
    expected_shared_area_m2: float | None = None


def build_empty_fixture() -> GeneratedFixture:
    model, _, building, storey = _base_model()
    return GeneratedFixture(
        content=model.to_string().encode("utf-8"),
        expected_space_count=0,
        expected_opening_count=0,
        storey_name=storey.Name,
        building_name=building.Name,
    )


def build_single_room_fixture() -> GeneratedFixture:
    model, body_context, building, storey = _base_model()
    room = _add_space(
        model,
        body_context,
        storey,
        name="Simple Room",
        placement=(5.0, 6.0, 0.0),
        represented=True,
    )
    return GeneratedFixture(
        content=model.to_string().encode("utf-8"),
        expected_space_count=1,
        expected_opening_count=0,
        represented_space_name=room.Name,
        storey_name=storey.Name,
        building_name=building.Name,
        space_names=(room.Name,),
    )


def build_shared_wall_fixture() -> GeneratedFixture:
    model, body_context, building, storey = _base_model()
    first_room = _add_space(
        model,
        body_context,
        storey,
        name="Room A",
        placement=(0.0, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    second_room = _add_space(
        model,
        body_context,
        storey,
        name="Room B",
        placement=(4.2, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    return GeneratedFixture(
        content=model.to_string().encode("utf-8"),
        expected_space_count=2,
        expected_opening_count=0,
        represented_space_name=first_room.Name,
        storey_name=storey.Name,
        building_name=building.Name,
        space_names=(first_room.Name, second_room.Name),
        expected_adjacency_pairs=((first_room.Name, second_room.Name),),
        expected_shared_area_m2=8.4,
    )
 
 
def build_separated_rooms_fixture() -> GeneratedFixture:
    model, body_context, building, storey = _base_model()
    first_room = _add_space(
        model,
        body_context,
        storey,
        name="Room A",
        placement=(0.0, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    second_room = _add_space(
        model,
        body_context,
        storey,
        name="Room B",
        placement=(4.6, 0.0, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=2.8,
    )
    return GeneratedFixture(
        content=model.to_string().encode("utf-8"),
        expected_space_count=2,
        expected_opening_count=0,
        represented_space_name=first_room.Name,
        storey_name=storey.Name,
        building_name=building.Name,
        space_names=(first_room.Name, second_room.Name),
    )


def build_corridor_room_fixture() -> GeneratedFixture:
    model, body_context, building, storey = _base_model()
    corridor = _add_space(
        model,
        body_context,
        storey,
        name="Corridor",
        placement=(0.0, 0.0, 0.0),
        represented=True,
        footprint=(8000.0, 2000.0),
        height=3.0,
    )
    touching_room = _add_space(
        model,
        body_context,
        storey,
        name="Touching Room",
        placement=(0.0, 2.5, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=3.0,
    )
    distant_room = _add_space(
        model,
        body_context,
        storey,
        name="Far Room",
        placement=(0.0, -4.1, 0.0),
        represented=True,
        footprint=(4000.0, 3000.0),
        height=3.0,
    )
    return GeneratedFixture(
        content=model.to_string().encode("utf-8"),
        expected_space_count=3,
        expected_opening_count=0,
        represented_space_name=corridor.Name,
        storey_name=storey.Name,
        building_name=building.Name,
        space_names=(corridor.Name, touching_room.Name, distant_room.Name),
        expected_adjacency_pairs=((corridor.Name, touching_room.Name),),
        expected_shared_area_m2=12.0,
    )


def build_extraction_fixture() -> GeneratedFixture:
    model, body_context, building, storey = _base_model()

    represented_space = _add_space(
        model,
        body_context,
        storey,
        name="Room With Geometry",
        placement=(5.0, 6.0, 0.0),
        represented=True,
    )
    missing_space = _add_space(
        model,
        body_context,
        storey,
        name="Room Missing Geometry",
        placement=(8.0, 2.0, 0.0),
        represented=False,
    )
    opening = _add_opening(
        model,
        body_context,
        storey,
        name="Opening 1",
        placement=(3.0, 1.0, 0.0),
    )

    return GeneratedFixture(
        content=model.to_string().encode("utf-8"),
        expected_space_count=2,
        expected_opening_count=1,
        represented_space_name=represented_space.Name,
        missing_space_name=missing_space.Name,
        opening_name=opening.Name,
        storey_name=storey.Name,
        building_name=building.Name,
        space_names=(represented_space.Name, missing_space.Name),
    )


def _add_space(
    model: Any,
    body_context: Any,
    storey: Any,
    *,
    name: str,
    placement: tuple[float, float, float],
    represented: bool,
    footprint: tuple[float, float] = (4.0, 3.0),
    height: float = 2.8,
) -> Any:
    space = run("root.create_entity", model, ifc_class="IfcSpace", name=name)
    run("aggregate.assign_object", model, products=[space], relating_object=storey)
    run(
        "geometry.edit_object_placement",
        model,
        product=space,
        matrix=_placement_matrix(*placement),
    )

    if represented:
        profile = model.create_entity(
            "IfcRectangleProfileDef",
            ProfileType="AREA",
            XDim=footprint[0],
            YDim=footprint[1],
        )
        representation = run(
            "geometry.add_profile_representation",
            model,
            context=body_context,
            profile=profile,
            depth=height,
        )
        run(
            "geometry.assign_representation",
            model,
            product=space,
            representation=representation,
        )

    return space


def _add_opening(
    model: Any,
    body_context: Any,
    storey: Any,
    *,
    name: str,
    placement: tuple[float, float, float],
) -> Any:
    opening = run("root.create_entity", model, ifc_class="IfcOpeningElement", name=name)
    run("spatial.assign_container", model, products=[opening], relating_structure=storey)
    run(
        "geometry.edit_object_placement",
        model,
        product=opening,
        matrix=_placement_matrix(*placement),
    )
    profile = model.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA",
        XDim=1.0,
        YDim=0.25,
    )
    representation = run(
        "geometry.add_profile_representation",
        model,
        context=body_context,
        profile=profile,
        depth=2.0,
    )
    run(
        "geometry.assign_representation",
        model,
        product=opening,
        representation=representation,
    )
    return opening


def _placement_matrix(x: float, y: float, z: float) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _base_model() -> tuple[Any, Any, Any, Any]:
    model = run("project.create_file", version="IFC4")
    project = run("root.create_entity", model, ifc_class="IfcProject", name="Fixture Project")
    run("unit.assign_unit", model)
    model_context = run("context.add_context", model, context_type="Model")
    body_context = run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_context,
    )

    site = run("root.create_entity", model, ifc_class="IfcSite", name="Fixture Site")
    building = run("root.create_entity", model, ifc_class="IfcBuilding", name="Building A")
    storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1")

    run("aggregate.assign_object", model, products=[site], relating_object=project)
    run("aggregate.assign_object", model, products=[building], relating_object=site)
    run("aggregate.assign_object", model, products=[storey], relating_object=building)

    run("geometry.edit_object_placement", model, product=site)
    run("geometry.edit_object_placement", model, product=building)
    run("geometry.edit_object_placement", model, product=storey)

    return model, body_context, building, storey
