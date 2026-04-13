from __future__ import annotations

from typing import Any

from app.mesh_normalizer import normalize_mesh


def test_cube_mesh_normalizes_to_positive_closed_solid() -> None:
    vertices, faces = build_cube_mesh()

    result = normalize_mesh(vertices, faces)

    assert result["valid"] is True
    assert result["closed"] is True
    assert result["outward_normals"] is True
    assert result["volume_m3"] > 0
    assert result["component_count"] == 1


def test_inverted_cube_winding_is_flipped_outward() -> None:
    vertices, faces = build_cube_mesh(invert_winding=True)

    result = normalize_mesh(vertices, faces)

    assert result["valid"] is True
    assert result["outward_normals"] is True
    assert result["volume_m3"] > 0
    assert any(action.startswith("flipped_component_winding") for action in result["repair_actions"])


def test_open_mesh_is_flagged_invalid_without_fake_repair() -> None:
    vertices, faces = build_cube_mesh(open_top=True)

    result = normalize_mesh(vertices, faces)

    assert result["valid"] is False
    assert result["closed"] is False
    assert result["outward_normals"] is False
    assert "Mesh is open" in result["reason"]


def test_disconnected_mesh_reports_multiple_components() -> None:
    vertices, faces = build_disconnected_cube_mesh()

    result = normalize_mesh(vertices, faces)

    assert result["valid"] is True
    assert result["component_count"] == 2
    assert len(result["components"]) == 2
    assert all(component["closed"] is True for component in result["components"])


def build_cube_mesh(*, invert_winding: bool = False, open_top: bool = False) -> tuple[list[list[float]], list[list[int]]]:
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [0, 7, 3],
        [0, 4, 7],
    ]

    if not open_top:
        faces.extend(
            [
                [3, 6, 2],
                [3, 7, 6],
            ]
        )

    if invert_winding:
        faces = [[face[0], face[2], face[1]] for face in faces]

    return vertices, faces


def build_disconnected_cube_mesh() -> tuple[list[list[float]], list[list[int]]]:
    left_vertices, left_faces = build_cube_mesh()
    right_vertices, right_faces = build_cube_mesh()
    translated_right_vertices: list[list[float]] = [[value + 3.0 if index == 0 else value for index, value in enumerate(vertex)] for vertex in right_vertices]

    vertex_offset = len(left_vertices)
    merged_vertices = left_vertices + translated_right_vertices
    merged_faces = left_faces + [[index + vertex_offset for index in face] for face in right_faces]
    return merged_vertices, merged_faces
