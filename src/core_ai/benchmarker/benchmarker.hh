#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include "json.hpp"

using namespace std::chrono;

struct QuantizedModelInfo {
    std::string modelId;
    std::string fileName;
    std::string downloadUrl; 
    std::string lastModified; 
};

struct benchmarker{
    std::vector<std::string> model_paths;
    std::string json_file_model_paths;
    std::vector<std::string> input_texts;

}