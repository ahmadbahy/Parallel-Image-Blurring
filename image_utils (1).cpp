#include "image_utils.h"
#include <algorithm>
#include <stdexcept>

cv::Mat load_color_image_or_fail(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + path);
    }

    return image;
}

void save_image_or_fail(const std::string& path, const cv::Mat& image) {
    bool ok = cv::imwrite(path, image);

    if (!ok) {
        throw std::runtime_error("Could not save image: " + path);
    }
}

cv::Mat extract_chunk_with_halo(
    const cv::Mat& image, 
    int output_start_row, 
    int output_rows, 
    int radius, 
    int& input_start_row, 
    int& top_halo_rows
) {
    int requested_start = output_start_row - radius; // real - halo 
    int requested_end = output_start_row + output_rows + radius; // real + halo

    // clip
    input_start_row = std::max(0, requested_start);
    int input_end_row = std::min(image.rows, requested_end);

    // real & n_halo
    top_halo_rows = output_start_row - input_start_row;
    int input_rows = input_end_row - input_start_row;

    // extract chunk [x, y, width, height]
    cv::Rect roi(0, input_start_row, image.cols, input_rows);
    return image(roi).clone();
}
