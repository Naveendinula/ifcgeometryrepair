#pragma once

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace worker_common {

using json = nlohmann::json;
using Vec3 = std::array<double, 3>;
using Face = std::array<int, 3>;

struct RawMesh {
  std::vector<Vec3> vertices;
  std::vector<Face> faces;
};

json read_json_file(const std::string& path);
void write_json_file(const std::string& path, const json& payload);
RawMesh parse_raw_mesh(const json& payload);
json mesh_to_json(const RawMesh& mesh);
double triangle_area_squared(const Vec3& a, const Vec3& b, const Vec3& c);

}  // namespace worker_common
