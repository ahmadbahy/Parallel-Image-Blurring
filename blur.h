#ifndef BLUR_H
#define BLUR_H

#include <opencv2/opencv.hpp>

enum class PaddingType {
    ZERO,
    MIRROR,
    NONE
};

long mirror_index(long index, long limit);

unsigned char get_pixel_value(
    const cv::Mat& img,
    int y,
    int x,
    int c,
    PaddingType padding,
    bool& valid
);

cv::Mat blur_serial(
    const cv::Mat& input,
    int radius,
    PaddingType padding
);

cv::Mat blur_openmp(
    const cv::Mat& input,
    int radius,
    PaddingType padding
);

cv::Mat blur_docker_chunk(
    const cv::Mat& chunk,
    int radius,
    int top_halo_rows,
    int output_rows,
    PaddingType padding
);

#endif