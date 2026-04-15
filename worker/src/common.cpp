#include "common.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace worker_common {

namespace {

Vec3 subtract(const Vec3& left, const Vec3& right) {
  return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

Vec3 cross(const Vec3& left, const Vec3& right) {
  return {
      left[1] * right[2] - left[2] * right[1],
      left[2] * right[0] - left[0] * right[2],
      left[0] * right[1] - left[1] * right[0],
  };
}

double norm_squared(const Vec3& value) {
  return (value[0] * value[0]) + (value[1] * value[1]) + (value[2] * value[2]);
}

}  // namespace

json read_json_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open request payload: " + path);
  }
  json payload;
  input >> payload;
  return payload;
}

void write_json_file(const std::string& path, const json& payload) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open result path: " + path);
  }
  output << payload.dump(2) << '\n';
}

RawMesh parse_raw_mesh(const json& payload) {
  RawMesh mesh;
  for (const auto& vertex : payload.at("vertices")) {
    mesh.vertices.push_back({vertex.at(0).get<double>(), vertex.at(1).get<double>(), vertex.at(2).get<double>()});
  }
  for (const auto& face : payload.at("faces")) {
    mesh.faces.push_back({face.at(0).get<int>(), face.at(1).get<int>(), face.at(2).get<int>()});
  }
  return mesh;
}

json mesh_to_json(const RawMesh& mesh) {
  return {
      {"vertices", mesh.vertices},
      {"faces", mesh.faces},
  };
}

double triangle_area_squared(const Vec3& a, const Vec3& b, const Vec3& c) {
  const Vec3 left = subtract(b, a);
  const Vec3 right = subtract(c, a);
  return norm_squared(cross(left, right)) * 0.25;
}

}  // namespace worker_common
