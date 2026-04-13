from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


WELD_EPSILON = 1e-7
AREA_EPSILON = 1e-12
VOLUME_EPSILON = 1e-12


def normalize_mesh(
    vertices: list[list[float]] | np.ndarray,
    faces: list[list[int]] | np.ndarray,
    *,
    weld_epsilon: float = WELD_EPSILON,
) -> dict[str, Any]:
    vertices_array = _as_vertex_array(vertices)
    faces_array = _as_face_array(faces)
    repair_actions: list[str] = []

    if len(vertices_array) == 0 or len(faces_array) == 0:
        return _empty_result("Mesh is empty", repair_actions)

    faces_array, dropped_invalid = _drop_invalid_faces(vertices_array, faces_array)
    if dropped_invalid:
        repair_actions.append(f"dropped_invalid_faces:{dropped_invalid}")

    vertices_array, faces_array, welded_vertices = _weld_vertices(vertices_array, faces_array, weld_epsilon)
    if welded_vertices:
        repair_actions.append(f"welded_vertices:{welded_vertices}")

    faces_array, dropped_degenerate = _drop_degenerate_faces(vertices_array, faces_array)
    if dropped_degenerate:
        repair_actions.append(f"dropped_degenerate_faces:{dropped_degenerate}")

    faces_array, dropped_duplicate_faces = _remove_duplicate_faces(faces_array)
    if dropped_duplicate_faces:
        repair_actions.append(f"dropped_duplicate_faces:{dropped_duplicate_faces}")

    if len(vertices_array) == 0 or len(faces_array) == 0:
        return _empty_result("Mesh is empty after cleanup", repair_actions)

    oriented_faces = faces_array.copy()
    components = _split_connected_components(oriented_faces)
    component_summaries: list[dict[str, Any]] = []

    if len(components) > 1:
        repair_actions.append(f"split_disconnected_components:{len(components)}")

    for component_index, face_indices in enumerate(components):
        component_faces = oriented_faces[face_indices]
        component_faces, orientation_conflicts = _orient_component_faces(component_faces)
        oriented_faces[face_indices] = component_faces
        if orientation_conflicts:
            repair_actions.append(f"orientation_conflicts:{component_index}:{orientation_conflicts}")

        component_closed, component_manifold = _analyze_topology(component_faces)
        component_volume = _signed_volume(vertices_array, component_faces)
        flipped = False
        if component_volume < -VOLUME_EPSILON:
            component_faces = component_faces[:, [0, 2, 1]]
            oriented_faces[face_indices] = component_faces
            component_volume = _signed_volume(vertices_array, component_faces)
            flipped = True
            repair_actions.append(f"flipped_component_winding:{component_index}")

        component_vertex_ids = np.unique(component_faces.reshape(-1))
        component_outward = component_closed and component_manifold and component_volume > VOLUME_EPSILON
        component_summaries.append(
            {
                "component_index": component_index,
                "face_count": int(len(component_faces)),
                "vertex_count": int(len(component_vertex_ids)),
                "closed": component_closed,
                "manifold": component_manifold,
                "outward_normals": component_outward,
                "volume_m3": float(max(component_volume, 0.0)),
                "flipped_winding": flipped,
            }
        )

    vertices_array, oriented_faces = _compact_vertices(vertices_array, oriented_faces)
    closed, manifold = _analyze_topology(oriented_faces)
    volume = _signed_volume(vertices_array, oriented_faces)
    if volume < -VOLUME_EPSILON:
        oriented_faces = oriented_faces[:, [0, 2, 1]]
        volume = _signed_volume(vertices_array, oriented_faces)
        repair_actions.append("flipped_mesh_winding")

    outward_normals = closed and manifold and volume > VOLUME_EPSILON
    valid = len(oriented_faces) > 0 and closed and manifold and volume > VOLUME_EPSILON
    reasons: list[str] = []
    if len(oriented_faces) == 0:
        reasons.append("Mesh is empty after cleanup")
    if not closed:
        reasons.append("Mesh is open")
    if not manifold:
        reasons.append("Mesh is non-manifold")
    if volume <= VOLUME_EPSILON:
        reasons.append("Non-positive volume")

    return {
        "mesh": {
            "vertices": vertices_array.tolist(),
            "faces": oriented_faces.tolist(),
        },
        "vertex_count": int(len(vertices_array)),
        "face_count": int(len(oriented_faces)),
        "component_count": int(len(component_summaries)),
        "components": component_summaries,
        "repair_actions": repair_actions,
        "closed": closed,
        "manifold": manifold,
        "outward_normals": outward_normals,
        "volume_m3": float(max(volume, 0.0)),
        "valid": valid,
        "reason": None if valid else "; ".join(reasons),
    }


def build_obj_text(meshes: list[dict[str, Any]]) -> str:
    lines = ["# IFC geometry repair normalized mesh export"]
    vertex_offset = 0

    for mesh in meshes:
        vertices = _as_vertex_array(mesh.get("vertices", []))
        faces = _as_face_array(mesh.get("faces", []))
        name = mesh.get("name") or "mesh"

        lines.append(f"o {name}")
        for vertex in vertices:
            lines.append(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}")

        for face in faces:
            a, b, c = face + vertex_offset + 1
            lines.append(f"f {a} {b} {c}")

        vertex_offset += len(vertices)

    lines.append("")
    return "\n".join(lines)


def _empty_result(reason: str, repair_actions: list[str]) -> dict[str, Any]:
    return {
        "mesh": None,
        "vertex_count": 0,
        "face_count": 0,
        "component_count": 0,
        "components": [],
        "repair_actions": repair_actions,
        "closed": False,
        "manifold": False,
        "outward_normals": False,
        "volume_m3": 0.0,
        "valid": False,
        "reason": reason,
    }


def _as_vertex_array(vertices: list[list[float]] | np.ndarray) -> np.ndarray:
    vertices_array = np.asarray(vertices, dtype=np.float64)
    if vertices_array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return vertices_array.reshape(-1, 3)


def _as_face_array(faces: list[list[int]] | np.ndarray) -> np.ndarray:
    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return faces_array.reshape(-1, 3)


def _drop_invalid_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    if len(faces) == 0:
        return faces, 0

    within_bounds = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    unique_indices = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    valid_mask = within_bounds & unique_indices
    dropped_count = int(len(faces) - np.count_nonzero(valid_mask))
    return faces[valid_mask], dropped_count


def _drop_degenerate_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    if len(faces) == 0:
        return faces, 0

    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    valid_mask = np.linalg.norm(cross_products, axis=1) > AREA_EPSILON
    dropped_count = int(len(faces) - np.count_nonzero(valid_mask))
    return faces[valid_mask], dropped_count


def _weld_vertices(vertices: np.ndarray, faces: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, int]:
    if len(vertices) == 0:
        return vertices, faces, 0

    quantized = np.round(vertices / epsilon).astype(np.int64)
    vertex_map: dict[tuple[int, int, int], int] = {}
    remapped_vertices: list[np.ndarray] = []
    remap_indices = np.empty(len(vertices), dtype=np.int64)

    for index, quantized_vertex in enumerate(quantized):
        key = tuple(int(value) for value in quantized_vertex)
        target_index = vertex_map.get(key)
        if target_index is None:
            target_index = len(remapped_vertices)
            vertex_map[key] = target_index
            remapped_vertices.append(vertices[index])
        remap_indices[index] = target_index

    remapped_faces = remap_indices[faces]
    welded_count = int(len(vertices) - len(remapped_vertices))
    return np.asarray(remapped_vertices, dtype=np.float64), remapped_faces, welded_count


def _remove_duplicate_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    if len(faces) == 0:
        return faces, 0

    seen: set[tuple[int, int, int]] = set()
    unique_faces: list[np.ndarray] = []
    dropped_count = 0

    for face in faces:
        key = tuple(sorted(int(value) for value in face))
        if key in seen:
            dropped_count += 1
            continue
        seen.add(key)
        unique_faces.append(face)

    return np.asarray(unique_faces, dtype=np.int64), dropped_count


def _split_connected_components(faces: np.ndarray) -> list[np.ndarray]:
    if len(faces) == 0:
        return []

    adjacency = _build_face_adjacency(faces)
    visited = np.zeros(len(faces), dtype=bool)
    components: list[np.ndarray] = []

    for face_index in range(len(faces)):
        if visited[face_index]:
            continue

        component: list[int] = []
        queue: deque[int] = deque([face_index])
        visited[face_index] = True

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor, _ in adjacency[current]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)

        components.append(np.asarray(component, dtype=np.int64))

    return components


def _orient_component_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    if len(faces) == 0:
        return faces, 0

    adjacency = _build_face_adjacency(faces)
    flips: dict[int, bool] = {}
    conflicts = 0

    for seed in range(len(faces)):
        if seed in flips:
            continue

        flips[seed] = False
        queue: deque[int] = deque([seed])

        while queue:
            current = queue.popleft()
            for neighbor, relation in adjacency[current]:
                expected_flip = flips[current] ^ bool(relation)
                if neighbor not in flips:
                    flips[neighbor] = expected_flip
                    queue.append(neighbor)
                    continue
                if flips[neighbor] != expected_flip:
                    conflicts += 1

    oriented_faces = faces.copy()
    for face_index, should_flip in flips.items():
        if should_flip:
            oriented_faces[face_index] = oriented_faces[face_index][[0, 2, 1]]

    return oriented_faces, conflicts


def _build_face_adjacency(faces: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    adjacency = {index: [] for index in range(len(faces))}
    edge_map: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}

    for face_index, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_key = tuple(sorted(edge))
            edge_map.setdefault(edge_key, []).append((face_index, edge))

    for references in edge_map.values():
        if len(references) < 2:
            continue

        for index, (left_face, left_direction) in enumerate(references):
            for right_face, right_direction in references[index + 1 :]:
                relation = 1 if left_direction == right_direction else 0
                adjacency[left_face].append((right_face, relation))
                adjacency[right_face].append((left_face, relation))

    return adjacency


def _analyze_topology(faces: np.ndarray) -> tuple[bool, bool]:
    if len(faces) == 0:
        return False, False

    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for edge in _face_edges(face):
            edge_key = tuple(sorted(edge))
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

    closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    manifold = bool(edge_counts) and all(count <= 2 for count in edge_counts.values())
    return closed, manifold


def _compact_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0 or len(faces) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    used_vertices = np.unique(faces.reshape(-1))
    remap = {int(old_index): new_index for new_index, old_index in enumerate(used_vertices)}
    compacted_vertices = vertices[used_vertices]
    compacted_faces = np.vectorize(lambda vertex_index: remap[int(vertex_index)], otypes=[np.int64])(faces)
    return compacted_vertices, compacted_faces


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0

    triangles = vertices[faces]
    tetrahedra = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return float(np.sum(tetrahedra) / 6.0)


def _face_edges(face: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a, b, c = (int(value) for value in face)
    return ((a, b), (b, c), (c, a))
