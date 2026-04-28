#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>

cv::Mat load_color_image_or_fail(const std::string& path);
void save_image_or_fail(const std::string& path, const cv::Mat& image);

cv::Mat extract_chunk_with_halo(
    const cv::Mat& image, 
    int output_start_row, 
    int output_rows, 
    int radius, 
    int& input_start_row, 
    int& top_halo_rows
);
#endif
