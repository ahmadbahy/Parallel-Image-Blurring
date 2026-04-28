#include "blur.h"
#include "image_utils.h"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

static PaddingType parse_padding(const std::string& text) {
    if (text == "zero") {
        return PaddingType::ZERO;
    }

    if (text == "mirror") {
        return PaddingType::MIRROR;
    }

    if (text == "none") {
        return PaddingType::NONE;
    }

    throw std::runtime_error("Invalid padding type in metadata: " + text);
}

static std::map<std::string, std::string> read_metadata(const std::string& meta_path) {
    std::ifstream file(meta_path);

    if (!file) {
        throw std::runtime_error("Could not open metadata file: " + meta_path);
    }

    std::map<std::string, std::string> meta;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        meta[key] = value;
    }

    return meta;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: worker <input_chunk_image> <output_chunk_image> <metadata_file>\n";
            return 1;
        }

        std::string input_chunk_path = argv[1];
        std::string output_chunk_path = argv[2];
        std::string metadata_path = argv[3];

        std::map<std::string, std::string> meta = read_metadata(metadata_path);

        int radius = std::stoi(meta.at("radius"));
        int top_halo_rows = std::stoi(meta.at("top_halo_rows"));
        int output_rows = std::stoi(meta.at("output_rows"));
        PaddingType padding = parse_padding(meta.at("padding"));

        cv::Mat chunk = load_color_image_or_fail(input_chunk_path);

        cv::Mat result = blur_docker_chunk(
            chunk,
            radius,
            top_halo_rows,
            output_rows,
            padding
        );

        save_image_or_fail(output_chunk_path, result);

        std::cout << "Worker finished: " << output_chunk_path << std::endl;
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Worker error: " << ex.what() << std::endl;
        return 2;
    }
}