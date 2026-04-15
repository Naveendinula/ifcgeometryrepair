#include "common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/boost/graph/helpers.h>

using worker_common::Face;
using worker_common::RawMesh;
using worker_common::Vec3;
using worker_common::json;
using worker_common::mesh_to_json;
using worker_common::parse_raw_mesh;
using worker_common::read_json_file;
using worker_common::triangle_area_squared;
using worker_common::write_json_file;

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using WrapMesh = CGAL::Surface_mesh<Point_3>;
using SoupFace = std::array<std::size_t, 3>;

constexpr int kContractVersion = 1;
constexpr double kAreaEpsilon = 1e-12;
constexpr char kBackend[] = "cpp-cgal-alpha-wrap";

struct ShellRequest {
  double alpha_m_effective = 0.0;
  double offset_m_effective = 0.0;
  std::vector<RawMesh> space_meshes;
};

json make_empty_shell_payload() {
  return mesh_to_json(RawMesh{});
}

json build_response_base(
    const std::optional<double> alpha_m_effective,
    const std::optional<double> offset_m_effective) {
  return {
      {"contract_version", kContractVersion},
      {"status", "failed"},
      {"backend", kBackend},
      {"reason", nullptr},
      {"alpha_m_effective", alpha_m_effective.value_or(0.0)},
      {"offset_m_effective", offset_m_effective.value_or(0.0)},
      {"shell_mesh", make_empty_shell_payload()},
  };
}

json build_failure_response(
    const std::string& reason,
    const std::optional<double> alpha_m_effective = std::nullopt,
    const std::optional<double> offset_m_effective = std::nullopt) {
  json response = build_response_base(alpha_m_effective, offset_m_effective);
  response["reason"] = reason;
  return response;
}

json build_failure_response_from_request(const json& request_payload, const std::string& reason) {
  return build_failure_response(
      reason,
      request_payload.contains("alpha_m_effective") ? std::optional<double>(request_payload.at("alpha_m_effective").get<double>())
                                                    : std::nullopt,
      request_payload.contains("offset_m_effective") ? std::optional<double>(request_payload.at("offset_m_effective").get<double>())
                                                     : std::nullopt);
}

RawMesh export_surface_mesh(const WrapMesh& mesh) {
  RawMesh exported;
  std::vector<int> vertex_indices(mesh.number_of_vertices(), -1);
  exported.vertices.reserve(mesh.number_of_vertices());

  for (const auto vertex : mesh.vertices()) {
    vertex_indices[static_cast<std::size_t>(vertex.idx())] = static_cast<int>(exported.vertices.size());
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
      indices.push_back(vertex_indices[static_cast<std::size_t>(vertex.idx())]);
    }
    if (indices.size() == 3) {
      exported.faces.push_back({indices[0], indices[1], indices[2]});
    }
  }
  return exported;
}

ShellRequest parse_request(const json& request_payload) {
  ShellRequest request;
  request.alpha_m_effective = request_payload.at("alpha_m_effective").get<double>();
  request.offset_m_effective = request_payload.at("offset_m_effective").get<double>();
  if (request.alpha_m_effective <= 0.0) {
    throw std::runtime_error("Alpha wrap alpha must be strictly positive.");
  }
  if (request.offset_m_effective <= 0.0) {
    throw std::runtime_error("Alpha wrap offset must be strictly positive.");
  }

  const auto& space_meshes_payload = request_payload.at("space_meshes");
  if (!space_meshes_payload.is_array() || space_meshes_payload.empty()) {
    throw std::runtime_error("No input space meshes were provided.");
  }

  request.space_meshes.reserve(space_meshes_payload.size());
  for (const auto& space_payload : space_meshes_payload) {
    request.space_meshes.push_back(parse_raw_mesh(space_payload.at("mesh")));
  }
  return request;
}

void build_triangle_soup(
    const ShellRequest& request,
    std::vector<Point_3>* points,
    std::vector<SoupFace>* faces,
    int* dropped_invalid_faces,
    int* dropped_degenerate_faces) {
  std::size_t vertex_offset = 0;
  for (const auto& mesh : request.space_meshes) {
    for (const auto& vertex : mesh.vertices) {
      points->emplace_back(vertex[0], vertex[1], vertex[2]);
    }
    for (const auto& face : mesh.faces) {
      const bool within_bounds = std::all_of(face.begin(), face.end(), [&](int index) {
        return index >= 0 && index < static_cast<int>(mesh.vertices.size());
      });
      const bool unique = face[0] != face[1] && face[1] != face[2] && face[0] != face[2];
      if (!within_bounds || !unique) {
        ++(*dropped_invalid_faces);
        continue;
      }

      const auto& a = mesh.vertices[static_cast<std::size_t>(face[0])];
      const auto& b = mesh.vertices[static_cast<std::size_t>(face[1])];
      const auto& c = mesh.vertices[static_cast<std::size_t>(face[2])];
      if (triangle_area_squared(a, b, c) <= kAreaEpsilon) {
        ++(*dropped_degenerate_faces);
        continue;
      }

      faces->push_back({
          vertex_offset + static_cast<std::size_t>(face[0]),
          vertex_offset + static_cast<std::size_t>(face[1]),
          vertex_offset + static_cast<std::size_t>(face[2]),
      });
    }
    vertex_offset += mesh.vertices.size();
  }
}

json process_request(const json& request_payload) {
  if (!request_payload.contains("contract_version") ||
      request_payload.at("contract_version").get<int>() != kContractVersion) {
    return build_failure_response_from_request(request_payload, "Unsupported request contract version.");
  }

  ShellRequest request;
  try {
    request = parse_request(request_payload);
  } catch (const std::exception& exc) {
    return build_failure_response_from_request(request_payload, exc.what());
  }

  std::vector<Point_3> points;
  std::vector<SoupFace> faces;
  int dropped_invalid_faces = 0;
  int dropped_degenerate_faces = 0;
  build_triangle_soup(request, &points, &faces, &dropped_invalid_faces, &dropped_degenerate_faces);

  if (points.empty() || faces.empty()) {
    std::string reason = "No valid input triangles were provided.";
    if (dropped_invalid_faces > 0 && dropped_degenerate_faces > 0) {
      reason = "No valid input triangles were provided after dropping invalid and degenerate faces.";
    } else if (dropped_invalid_faces > 0) {
      reason = "No valid input triangles were provided after dropping invalid faces.";
    } else if (dropped_degenerate_faces > 0) {
      reason = "No valid input triangles were provided after dropping degenerate faces.";
    }
    return build_failure_response(reason, request.alpha_m_effective, request.offset_m_effective);
  }

  WrapMesh wrap;
  const auto started_at = std::chrono::steady_clock::now();
  CGAL::alpha_wrap_3(points, faces, request.alpha_m_effective, request.offset_m_effective, wrap);
  const auto finished_at = std::chrono::steady_clock::now();
  const double generation_time_ms = std::chrono::duration<double, std::milli>(finished_at - started_at).count();

  RawMesh shell_mesh = export_surface_mesh(wrap);
  if (shell_mesh.vertices.empty() || shell_mesh.faces.empty()) {
    return build_failure_response(
        "Alpha wrap produced an empty shell mesh.",
        request.alpha_m_effective,
        request.offset_m_effective);
  }

  json response = build_response_base(request.alpha_m_effective, request.offset_m_effective);
  response["status"] = "ok";
  response["generation_time_ms"] = generation_time_ms;
  response["shell_mesh"] = mesh_to_json(shell_mesh);
  return response;
}

RawMesh make_cube_mesh(const Vec3& origin, double size) {
  const double x = origin[0];
  const double y = origin[1];
  const double z = origin[2];
  return RawMesh{
      {
          Vec3{x, y, z},
          Vec3{x + size, y, z},
          Vec3{x + size, y + size, z},
          Vec3{x, y + size, z},
          Vec3{x, y, z + size},
          Vec3{x + size, y, z + size},
          Vec3{x + size, y + size, z + size},
          Vec3{x, y + size, z + size},
      },
      {
          Face{0, 2, 1},
          Face{0, 3, 2},
          Face{4, 5, 6},
          Face{4, 6, 7},
          Face{0, 1, 5},
          Face{0, 5, 4},
          Face{1, 2, 6},
          Face{1, 6, 5},
          Face{2, 3, 7},
          Face{2, 7, 6},
          Face{3, 0, 4},
          Face{3, 4, 7},
      },
  };
}

json build_self_test_request(
    const std::vector<RawMesh>& space_meshes,
    int contract_version = kContractVersion,
    double alpha_m_effective = 1.0,
    double offset_m_effective = 0.05) {
  json request = {
      {"contract_version", contract_version},
      {"job_id", "self-test"},
      {"unit", "meter"},
      {"mode_requested", "alpha_wrap"},
      {"epsilon", 1e-3},
      {"alpha_m_requested", alpha_m_effective},
      {"alpha_m_effective", alpha_m_effective},
      {"offset_m_requested", offset_m_effective},
      {"offset_m_effective", offset_m_effective},
      {"space_meshes", json::array()},
      {"candidate_surfaces", json::array()},
  };
  for (std::size_t index = 0; index < space_meshes.size(); ++index) {
    request["space_meshes"].push_back(
        {
            {"global_id", "space-" + std::to_string(index)},
            {"express_id", static_cast<int>(index + 1)},
            {"name", "space-" + std::to_string(index)},
            {"mesh", mesh_to_json(space_meshes[index])},
        });
  }
  return request;
}

int expect_success(const std::string& test_name, const json& response) {
  if (response.at("status") != "ok") {
    std::cerr << "self-test failed: " << test_name << " expected success but got failure\n";
    if (response.contains("reason") && !response.at("reason").is_null()) {
      std::cerr << "reason: " << response.at("reason").get<std::string>() << "\n";
    }
    return 1;
  }
  const auto& shell_mesh = response.at("shell_mesh");
  if (shell_mesh.at("vertices").empty() || shell_mesh.at("faces").empty()) {
    std::cerr << "self-test failed: " << test_name << " produced an empty shell mesh\n";
    return 1;
  }
  return 0;
}

int expect_failure(const std::string& test_name, const json& response, const std::string& reason_substring) {
  if (response.at("status") == "ok") {
    std::cerr << "self-test failed: " << test_name << " expected failure but got success\n";
    return 1;
  }
  const std::string reason = response.value("reason", "");
  if (reason.find(reason_substring) == std::string::npos) {
    std::cerr << "self-test failed: " << test_name << " missing expected reason substring '" << reason_substring << "'\n";
    std::cerr << "reason: " << reason << "\n";
    return 1;
  }
  return 0;
}

int run_self_test(const std::string& name) {
  if (name == "closed_cube") {
    return expect_success(name, process_request(build_self_test_request({make_cube_mesh({0.0, 0.0, 0.0}, 1.0)})));
  }
  if (name == "combined_rooms") {
    return expect_success(
        name,
        process_request(build_self_test_request(
            {make_cube_mesh({0.0, 0.0, 0.0}, 1.0), make_cube_mesh({1.5, 0.0, 0.0}, 1.0)})));
  }
  if (name == "invalid_contract") {
    return expect_failure(
        name,
        process_request(build_self_test_request({make_cube_mesh({0.0, 0.0, 0.0}, 1.0)}, 99)),
        "Unsupported request contract version");
  }
  if (name == "degenerate_mesh") {
    RawMesh degenerate;
    degenerate.vertices = {Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}, Vec3{2.0, 0.0, 0.0}};
    degenerate.faces = {Face{0, 1, 2}};
    return expect_failure(
        name,
        process_request(build_self_test_request({degenerate})),
        "No valid input triangles were provided");
  }

  std::cerr << "unknown self-test: " << name << "\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::string> result_path;
  try {
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return run_self_test(argv[2]);
    }

    if (argc != 3) {
      std::cerr << "usage: shell_worker <request.json> <result.json>\n";
      std::cerr << "   or: shell_worker --self-test <name>\n";
      return 2;
    }

    const std::string request_path = argv[1];
    result_path = argv[2];
    const json request_payload = read_json_file(request_path);
    write_json_file(*result_path, process_request(request_payload));
    return 0;
  } catch (const std::exception& exc) {
    if (result_path) {
      try {
        write_json_file(*result_path, build_failure_response(exc.what()));
        return 0;
      } catch (const std::exception&) {
      }
    }
    std::cerr << exc.what() << '\n';
    return 1;
  }
}
