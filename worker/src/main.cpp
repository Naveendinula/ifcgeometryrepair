#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/boost/graph/helpers.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using Polyhedron = CGAL::Polyhedron_3<Kernel>;
using Nef_polyhedron = CGAL::Nef_polyhedron_3<Kernel>;
using worker_common::Face;
using worker_common::json;
using worker_common::mesh_to_json;
using worker_common::parse_raw_mesh;
using worker_common::RawMesh;
using worker_common::read_json_file;
using worker_common::triangle_area_squared;
using worker_common::Vec3;
using worker_common::write_json_file;

namespace {

constexpr int kContractVersion = 2;
constexpr double kWeldEpsilon = 1e-7;
constexpr double kAreaEpsilon = 1e-12;
constexpr double kVolumeEpsilon = 1e-12;

struct SpaceRequest {
  std::string object_name;
  std::optional<std::string> global_id;
  int express_id = 0;
  std::optional<std::string> name;
  RawMesh mesh;
};

struct ComponentSummary {
  int component_index = 0;
  int face_count = 0;
  int vertex_count = 0;
  bool closed = false;
  bool manifold = false;
  bool outward_normals = false;
  double volume_m3 = 0.0;
  bool flipped_winding = false;
};

struct RepairResult {
  std::optional<std::string> global_id;
  int express_id = 0;
  std::optional<std::string> name;
  RawMesh mesh;
  int vertex_count = 0;
  int face_count = 0;
  int component_count = 0;
  std::vector<ComponentSummary> components;
  std::vector<std::string> repair_actions;
  std::string repair_backend = "cpp-cgal";
  std::string repair_status = "exact_repaired";
  std::optional<std::string> repair_reason;
  bool closed = false;
  bool manifold = false;
  bool outward_normals = false;
  double volume_m3 = 0.0;
  bool valid = false;
  std::optional<std::string> reason;
};

struct QuantizedVertexKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const QuantizedVertexKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct QuantizedVertexKeyHash {
  std::size_t operator()(const QuantizedVertexKey& key) const {
    std::size_t seed = 0;
    seed ^= std::hash<std::int64_t>{}(key.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct EdgeKeyHash {
  std::size_t operator()(const std::pair<int, int>& key) const {
    std::size_t seed = 0;
    seed ^= std::hash<int>{}(key.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(key.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

std::vector<std::pair<int, int>> face_edges(const Face& face) {
  return {
      {face[0], face[1]},
      {face[1], face[2]},
      {face[2], face[0]},
  };
}

std::string join_reasons(const std::vector<std::string>& reasons) {
  std::ostringstream stream;
  bool first = true;
  for (const auto& reason : reasons) {
    if (reason.empty()) {
      continue;
    }
    if (!first) {
      stream << "; ";
    }
    first = false;
    stream << reason;
  }
  return stream.str();
}

SpaceRequest parse_space_request(const json& payload) {
  SpaceRequest request;
  request.object_name = payload.at("object_name").get<std::string>();
  if (!payload["global_id"].is_null()) {
    request.global_id = payload.at("global_id").get<std::string>();
  }
  request.express_id = payload.at("express_id").get<int>();
  if (payload.contains("name") && !payload["name"].is_null()) {
    request.name = payload.at("name").get<std::string>();
  }
  request.mesh = parse_raw_mesh(payload.at("mesh"));
  return request;
}

void drop_invalid_faces(RawMesh* mesh, std::vector<std::string>* actions) {
  std::vector<Face> cleaned;
  int dropped = 0;
  for (const auto& face : mesh->faces) {
    const bool within_bounds = std::all_of(face.begin(), face.end(), [&](int index) {
      return index >= 0 && index < static_cast<int>(mesh->vertices.size());
    });
    const bool unique = face[0] != face[1] && face[1] != face[2] && face[0] != face[2];
    if (!within_bounds || !unique) {
      ++dropped;
      continue;
    }
    cleaned.push_back(face);
  }
  mesh->faces = std::move(cleaned);
  if (dropped > 0) {
    actions->push_back("dropped_invalid_faces:" + std::to_string(dropped));
  }
}

void weld_vertices(RawMesh* mesh, std::vector<std::string>* actions) {
  std::unordered_map<QuantizedVertexKey, int, QuantizedVertexKeyHash> index_by_key;
  std::vector<Vec3> welded_vertices;
  std::vector<int> remap(mesh->vertices.size(), -1);

  for (std::size_t index = 0; index < mesh->vertices.size(); ++index) {
    const auto& vertex = mesh->vertices[index];
    QuantizedVertexKey key{
        static_cast<std::int64_t>(std::llround(vertex[0] / kWeldEpsilon)),
        static_cast<std::int64_t>(std::llround(vertex[1] / kWeldEpsilon)),
        static_cast<std::int64_t>(std::llround(vertex[2] / kWeldEpsilon)),
    };
    const auto it = index_by_key.find(key);
    if (it != index_by_key.end()) {
      remap[index] = it->second;
      continue;
    }
    const int new_index = static_cast<int>(welded_vertices.size());
    index_by_key.emplace(key, new_index);
    welded_vertices.push_back(vertex);
    remap[index] = new_index;
  }

  for (auto& face : mesh->faces) {
    face = {remap[static_cast<std::size_t>(face[0])], remap[static_cast<std::size_t>(face[1])], remap[static_cast<std::size_t>(face[2])]};
  }

  const int welded = static_cast<int>(mesh->vertices.size() - welded_vertices.size());
  mesh->vertices = std::move(welded_vertices);
  if (welded > 0) {
    actions->push_back("welded_vertices:" + std::to_string(welded));
  }
}

void drop_degenerate_faces(RawMesh* mesh, std::vector<std::string>* actions) {
  std::vector<Face> cleaned;
  int dropped = 0;
  for (const auto& face : mesh->faces) {
    const auto& a = mesh->vertices[static_cast<std::size_t>(face[0])];
    const auto& b = mesh->vertices[static_cast<std::size_t>(face[1])];
    const auto& c = mesh->vertices[static_cast<std::size_t>(face[2])];
    if (triangle_area_squared(a, b, c) <= kAreaEpsilon) {
      ++dropped;
      continue;
    }
    cleaned.push_back(face);
  }
  mesh->faces = std::move(cleaned);
  if (dropped > 0) {
    actions->push_back("dropped_degenerate_faces:" + std::to_string(dropped));
  }
}

void remove_duplicate_faces(RawMesh* mesh, std::vector<std::string>* actions) {
  std::set<std::tuple<int, int, int>> seen;
  std::vector<Face> unique_faces;
  int dropped = 0;
  for (const auto& face : mesh->faces) {
    std::array<int, 3> sorted = face;
    std::sort(sorted.begin(), sorted.end());
    const auto [it, inserted] = seen.emplace(sorted[0], sorted[1], sorted[2]);
    (void)it;
    if (!inserted) {
      ++dropped;
      continue;
    }
    unique_faces.push_back(face);
  }
  mesh->faces = std::move(unique_faces);
  if (dropped > 0) {
    actions->push_back("dropped_duplicate_faces:" + std::to_string(dropped));
  }
}

void compact_vertices(RawMesh* mesh) {
  std::vector<int> used(mesh->vertices.size(), 0);
  for (const auto& face : mesh->faces) {
    used[static_cast<std::size_t>(face[0])] = 1;
    used[static_cast<std::size_t>(face[1])] = 1;
    used[static_cast<std::size_t>(face[2])] = 1;
  }

  std::vector<Vec3> compacted_vertices;
  std::vector<int> remap(mesh->vertices.size(), -1);
  for (std::size_t index = 0; index < mesh->vertices.size(); ++index) {
    if (!used[index]) {
      continue;
    }
    remap[index] = static_cast<int>(compacted_vertices.size());
    compacted_vertices.push_back(mesh->vertices[index]);
  }
  for (auto& face : mesh->faces) {
    face = {remap[static_cast<std::size_t>(face[0])], remap[static_cast<std::size_t>(face[1])], remap[static_cast<std::size_t>(face[2])]};
  }
  mesh->vertices = std::move(compacted_vertices);
}

bool try_regularize_with_nef(Surface_mesh* mesh, std::vector<std::string>* actions, std::string* error) {
  try {
    Polyhedron polyhedron;
    CGAL::copy_face_graph(*mesh, polyhedron);
    if (polyhedron.empty()) {
      *error = "Nef regularization received an empty polyhedron.";
      return false;
    }

    Nef_polyhedron nef(polyhedron);
    Nef_polyhedron regularized = nef.regularization();
    Polyhedron repaired_polyhedron;
    regularized.convert_to_polyhedron(repaired_polyhedron);
    if (repaired_polyhedron.empty()) {
      *error = "Nef regularization produced an empty polyhedron.";
      return false;
    }

    Surface_mesh repaired_mesh;
    CGAL::copy_face_graph(repaired_polyhedron, repaired_mesh);
    *mesh = std::move(repaired_mesh);
    actions->push_back("cgal_nef_regularization");
    return true;
  } catch (const std::exception& exc) {
    *error = exc.what();
    return false;
  }
}

RawMesh export_surface_mesh(const Surface_mesh& mesh) {
  RawMesh exported;
  std::unordered_map<std::size_t, int> vertex_indices;
  exported.vertices.reserve(mesh.number_of_vertices());

  for (const auto vertex : mesh.vertices()) {
    vertex_indices.emplace(vertex.idx(), static_cast<int>(exported.vertices.size()));
    const auto& point = mesh.point(vertex);
    exported.vertices.push_back({
        CGAL::to_double(point.x()),
        CGAL::to_double(point.y()),
        CGAL::to_double(point.z()),
    });
  }

  for (const auto face : mesh.faces()) {
    std::vector<int> indices;
    for (const auto vertex : CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
      indices.push_back(vertex_indices.at(vertex.idx()));
    }
    if (indices.size() == 3) {
      exported.faces.push_back({indices[0], indices[1], indices[2]});
    }
  }
  return exported;
}

double signed_volume(const RawMesh& mesh) {
  double volume = 0.0;
  for (const auto& face : mesh.faces) {
    const auto& a = mesh.vertices[static_cast<std::size_t>(face[0])];
    const auto& b = mesh.vertices[static_cast<std::size_t>(face[1])];
    const auto& c = mesh.vertices[static_cast<std::size_t>(face[2])];
    volume += a[0] * ((b[1] * c[2]) - (b[2] * c[1]));
    volume -= a[1] * ((b[0] * c[2]) - (b[2] * c[0]));
    volume += a[2] * ((b[0] * c[1]) - (b[1] * c[0]));
  }
  return volume / 6.0;
}

std::tuple<bool, bool, int, std::vector<ComponentSummary>> analyze_topology(const RawMesh& mesh) {
  std::unordered_map<std::pair<int, int>, int, EdgeKeyHash> edge_counts;
  std::unordered_map<std::pair<int, int>, std::vector<int>, EdgeKeyHash> edge_faces;
  std::vector<std::unordered_set<int>> adjacency(mesh.faces.size());

  for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
    for (const auto& edge : face_edges(mesh.faces[face_index])) {
      const auto key = std::minmax(edge.first, edge.second);
      ++edge_counts[key];
      auto& references = edge_faces[key];
      for (const int other_face : references) {
        adjacency[face_index].insert(other_face);
        adjacency[static_cast<std::size_t>(other_face)].insert(static_cast<int>(face_index));
      }
      references.push_back(static_cast<int>(face_index));
    }
  }

  const bool closed = !edge_counts.empty() &&
                      std::all_of(edge_counts.begin(), edge_counts.end(), [](const auto& item) {
                        return item.second == 2;
                      });
  const bool manifold = !edge_counts.empty() &&
                        std::all_of(edge_counts.begin(), edge_counts.end(), [](const auto& item) {
                          return item.second <= 2;
                        });

  std::vector<int> visited(mesh.faces.size(), 0);
  std::vector<ComponentSummary> components;
  int component_count = 0;
  for (std::size_t seed = 0; seed < mesh.faces.size(); ++seed) {
    if (visited[seed] != 0) {
      continue;
    }
    std::queue<int> queue;
    queue.push(static_cast<int>(seed));
    visited[seed] = 1;
    std::vector<int> component_faces;
    std::unordered_set<int> component_vertices;
    while (!queue.empty()) {
      const int current = queue.front();
      queue.pop();
      component_faces.push_back(current);
      const auto& face = mesh.faces[static_cast<std::size_t>(current)];
      component_vertices.insert(face[0]);
      component_vertices.insert(face[1]);
      component_vertices.insert(face[2]);
      for (const int neighbor : adjacency[static_cast<std::size_t>(current)]) {
        if (visited[static_cast<std::size_t>(neighbor)] != 0) {
          continue;
        }
        visited[static_cast<std::size_t>(neighbor)] = 1;
        queue.push(neighbor);
      }
    }

    RawMesh component_mesh;
    component_mesh.vertices = mesh.vertices;
    for (const int face_index : component_faces) {
      component_mesh.faces.push_back(mesh.faces[static_cast<std::size_t>(face_index)]);
    }
    const double component_volume = std::abs(signed_volume(component_mesh));
    components.push_back(
        ComponentSummary{
            component_count,
            static_cast<int>(component_faces.size()),
            static_cast<int>(component_vertices.size()),
            closed,
            manifold,
            closed && manifold && component_volume > kVolumeEpsilon,
            component_volume,
            false,
        });
    ++component_count;
  }

  return {closed, manifold, component_count, components};
}

RepairResult build_failure_result(
    const SpaceRequest& request,
    const std::vector<std::string>& actions,
    const std::string& reason) {
  RepairResult result;
  result.global_id = request.global_id;
  result.express_id = request.express_id;
  result.name = request.name;
  result.repair_actions = actions;
  result.repair_status = "exact_repaired";
  result.repair_reason = reason;
  result.valid = false;
  result.reason = reason;
  return result;
}

RepairResult repair_space(const SpaceRequest& request) {
  RawMesh working = request.mesh;
  std::vector<std::string> actions;

  if (working.vertices.empty() || working.faces.empty()) {
    return build_failure_result(request, actions, "Mesh is empty.");
  }

  drop_invalid_faces(&working, &actions);
  weld_vertices(&working, &actions);
  drop_degenerate_faces(&working, &actions);
  remove_duplicate_faces(&working, &actions);
  compact_vertices(&working);

  if (working.vertices.empty() || working.faces.empty()) {
    return build_failure_result(request, actions, "Mesh is empty after cleanup.");
  }

  std::vector<Point_3> points;
  points.reserve(working.vertices.size());
  for (const auto& vertex : working.vertices) {
    points.emplace_back(vertex[0], vertex[1], vertex[2]);
  }

  std::vector<std::vector<std::size_t>> polygons;
  polygons.reserve(working.faces.size());
  for (const auto& face : working.faces) {
    polygons.push_back(
        {static_cast<std::size_t>(face[0]), static_cast<std::size_t>(face[1]), static_cast<std::size_t>(face[2])});
  }

  Surface_mesh mesh;
  try {
    PMP::orient_polygon_soup(points, polygons);
    PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    PMP::triangulate_faces(mesh);
    PMP::stitch_borders(mesh);
    PMP::remove_isolated_vertices(mesh);
  } catch (const std::exception& exc) {
    return build_failure_result(request, actions, std::string("CGAL polygon soup conversion failed: ") + exc.what());
  }

  if (mesh.number_of_faces() == 0) {
    return build_failure_result(request, actions, "CGAL polygon soup conversion produced no faces.");
  }

  if (CGAL::is_closed(mesh)) {
    std::string nef_error;
    Surface_mesh regularized = mesh;
    if (try_regularize_with_nef(&regularized, &actions, &nef_error)) {
      mesh = std::move(regularized);
      PMP::triangulate_faces(mesh);
      PMP::remove_isolated_vertices(mesh);
    } else if (!nef_error.empty()) {
      actions.push_back("cgal_nef_regularization_failed");
    }

    PMP::orient_to_bound_a_volume(mesh);
  }

  const bool self_intersects = PMP::does_self_intersect(mesh);
  RawMesh exported = export_surface_mesh(mesh);
  const auto [closed, manifold, component_count, components] = analyze_topology(exported);
  const double volume = std::abs(signed_volume(exported));

  std::vector<std::string> reasons;
  if (exported.faces.empty()) {
    reasons.push_back("Mesh is empty after repair.");
  }
  if (!closed) {
    reasons.push_back("Mesh is open.");
  }
  if (!manifold) {
    reasons.push_back("Mesh is non-manifold.");
  }
  if (self_intersects) {
    reasons.push_back("Mesh self-intersects.");
  }
  if (volume <= kVolumeEpsilon) {
    reasons.push_back("Non-positive volume.");
  }

  RepairResult result;
  result.global_id = request.global_id;
  result.express_id = request.express_id;
  result.name = request.name;
  result.mesh = std::move(exported);
  result.vertex_count = static_cast<int>(result.mesh.vertices.size());
  result.face_count = static_cast<int>(result.mesh.faces.size());
  result.component_count = component_count;
  result.components = components;
  result.repair_actions = actions;
  result.closed = closed;
  result.manifold = manifold;
  result.outward_normals = closed && manifold && volume > kVolumeEpsilon && !self_intersects;
  result.volume_m3 = volume;
  result.valid = reasons.empty();
  result.repair_status = result.valid && actions.empty() ? "exact_passthrough" : "exact_repaired";
  if (!result.valid) {
    const std::string reason = join_reasons(reasons);
    result.repair_reason = reason;
    result.reason = reason;
  }
  return result;
}

json component_to_json(const ComponentSummary& component) {
  return {
      {"component_index", component.component_index},
      {"face_count", component.face_count},
      {"vertex_count", component.vertex_count},
      {"closed", component.closed},
      {"manifold", component.manifold},
      {"outward_normals", component.outward_normals},
      {"volume_m3", component.volume_m3},
      {"flipped_winding", component.flipped_winding},
  };
}

json result_to_json(const RepairResult& result) {
  json payload = {
      {"global_id", result.global_id ? json(*result.global_id) : json(nullptr)},
      {"express_id", result.express_id},
      {"name", result.name ? json(*result.name) : json(nullptr)},
      {"mesh", mesh_to_json(result.mesh)},
      {"vertex_count", result.vertex_count},
      {"face_count", result.face_count},
      {"component_count", result.component_count},
      {"components", json::array()},
      {"repair_actions", result.repair_actions},
      {"repair_backend", result.repair_backend},
      {"repair_status", result.repair_status},
      {"repair_reason", result.repair_reason ? json(*result.repair_reason) : json(nullptr)},
      {"closed", result.closed},
      {"manifold", result.manifold},
      {"outward_normals", result.outward_normals},
      {"volume_m3", result.volume_m3},
      {"valid", result.valid},
      {"reason", result.reason ? json(*result.reason) : json(nullptr)},
  };
  for (const auto& component : result.components) {
    payload["components"].push_back(component_to_json(component));
  }
  return payload;
}

json build_success_response(const json& request_payload) {
  json response = {
      {"contract_version", kContractVersion},
      {"status", "ok"},
      {"worker_backend", "cpp-cgal"},
      {"reason", nullptr},
      {"spaces", json::array()},
  };
  for (const auto& space_payload : request_payload.at("spaces")) {
    const SpaceRequest request = parse_space_request(space_payload);
    response["spaces"].push_back(result_to_json(repair_space(request)));
  }
  return response;
}

SpaceRequest make_cube_request(const std::string& name, bool invert_winding, bool open_top, bool duplicate_face, bool degenerate_face) {
  SpaceRequest request;
  request.object_name = name;
  request.global_id = name;
  request.express_id = 1;
  request.name = name;
  request.mesh.vertices = {
      Vec3{0.0, 0.0, 0.0},
      Vec3{1.0, 0.0, 0.0},
      Vec3{1.0, 1.0, 0.0},
      Vec3{0.0, 1.0, 0.0},
      Vec3{0.0, 0.0, 1.0},
      Vec3{1.0, 0.0, 1.0},
      Vec3{1.0, 1.0, 1.0},
      Vec3{0.0, 1.0, 1.0},
  };
  request.mesh.faces = {
      Face{0, 2, 1},
      Face{0, 3, 2},
      Face{4, 5, 6},
      Face{4, 6, 7},
      Face{0, 1, 5},
      Face{0, 5, 4},
      Face{1, 2, 6},
      Face{1, 6, 5},
      Face{0, 7, 3},
      Face{0, 4, 7},
  };
  if (!open_top) {
    request.mesh.faces.push_back(Face{3, 6, 2});
    request.mesh.faces.push_back(Face{3, 7, 6});
  }
  if (duplicate_face) {
    request.mesh.faces.push_back(Face{0, 2, 1});
  }
  if (degenerate_face) {
    request.mesh.faces.push_back(Face{0, 0, 1});
  }
  if (invert_winding) {
    for (auto& face : request.mesh.faces) {
      std::swap(face[1], face[2]);
    }
  }
  return request;
}

SpaceRequest make_self_intersecting_request() {
  SpaceRequest request = make_cube_request("self_intersection", false, false, false, false);
  SpaceRequest offset = make_cube_request("offset", false, false, false, false);
  for (auto& vertex : offset.mesh.vertices) {
    vertex[0] += 0.6;
  }
  const int vertex_offset = static_cast<int>(request.mesh.vertices.size());
  request.mesh.vertices.insert(request.mesh.vertices.end(), offset.mesh.vertices.begin(), offset.mesh.vertices.end());
  for (const auto& face : offset.mesh.faces) {
    request.mesh.faces.push_back(Face{face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset});
  }
  request.object_name = "self_intersection";
  request.global_id = "self_intersection";
  request.express_id = 2;
  request.name = "self_intersection";
  return request;
}

int run_self_test(const std::string& name) {
  auto expect = [&](const std::string& test_name, const RepairResult& result, bool should_be_valid) -> int {
    if (result.valid != should_be_valid) {
      std::cerr << "self-test failed: " << test_name << " expected valid=" << should_be_valid
                << " actual=" << result.valid << "\n";
      if (result.reason) {
        std::cerr << "reason: " << *result.reason << "\n";
      }
      return 1;
    }
    return 0;
  };

  if (name == "closed_cube") {
    return expect(name, repair_space(make_cube_request(name, false, false, false, false)), true);
  }
  if (name == "inverted_cube") {
    return expect(name, repair_space(make_cube_request(name, true, false, false, false)), true);
  }
  if (name == "duplicate_and_degenerate") {
    return expect(name, repair_space(make_cube_request(name, false, false, true, true)), true);
  }
  if (name == "open_mesh") {
    return expect(name, repair_space(make_cube_request(name, false, true, false, false)), false);
  }
  if (name == "self_intersection") {
    return expect(name, repair_space(make_self_intersecting_request()), false);
  }

  std::cerr << "unknown self-test: " << name << "\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return run_self_test(argv[2]);
    }

    if (argc != 3) {
      std::cerr << "usage: geometry_worker <request.json> <result.json>\n";
      std::cerr << "   or: geometry_worker --self-test <name>\n";
      return 2;
    }

    const std::string request_path = argv[1];
    const std::string result_path = argv[2];
    const json request_payload = read_json_file(request_path);
    if (request_payload.at("contract_version").get<int>() != kContractVersion) {
      throw std::runtime_error("unsupported repair contract version");
    }
    write_json_file(result_path, build_success_response(request_payload));
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << '\n';
    return 1;
  }
}
