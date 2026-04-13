#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: geometry_worker <request.json> <result.json>\n";
    return 2;
  }

  const std::string request_path = argv[1];
  const std::string result_path = argv[2];

  std::ifstream request_stream(request_path);
  if (!request_stream) {
    std::cerr << "could not open request payload: " << request_path << "\n";
    return 3;
  }

  std::ofstream result_stream(result_path);
  if (!result_stream) {
    std::cerr << "could not open result path: " << result_path << "\n";
    return 4;
  }

  result_stream << "{\n"
                << "  \"status\": \"scaffold_only\",\n"
                << "  \"message\": \"C++ geometry worker contract scaffold is present, but mesh normalization is still handled by the Python fallback in this phase.\"\n"
                << "}\n";

  return 0;
}
