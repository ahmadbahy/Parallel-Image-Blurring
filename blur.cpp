#include "blur.h"

// ---------------------------------------------------------------------------------------
// Utils

/*
 * Original
 * n = 3*3
 * 1 2 3
 * 4 5 6
 * 7 8 9
 *
 * Mirrored: maps an outside index back into [0, limit - 1]
 * if (idx is -) --> new_idx = -idx
 * if (idx >= n) --> new_idx = 2n - idx - 2
 * if (idx < n && idx >= 0) --> the original value
 * new_val = (new_idx_row, new_idx_col)
 *
 * -- indices
 * (-1, -1) (-1, 0) (-1, 1) (-1, 2) (-1, 3)
 * (0, -1)  (0, 0)  (0, 1) ....
 * (1, -1)
 * ..
 *
 * -- values
 * 5 4 5 6 5  
 * 2 1 2 3 2
 * 5 4 5 6 5
 * 8 7 8 9 8
 * 5 4 5 6 5
 */


long mirror_index(long index, long limit) {
    if (limit <= 1) {
        return 0;
    }

    while (index < 0 || index >= limit) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * limit - index - 2;
        }
    }

    return index;
}

unsigned char get_pixel_value(
    const cv::Mat& img,
    int y,
    int x,
    int c,
    PaddingType padding,
    bool& valid
) {
    
    bool inside =
        y >= 0 && y < img.rows &&
        x >= 0 && x < img.cols;

   
    if (inside) {
        valid = true;
        return img.at<cv::Vec3b>(y, x)[c];
    }


    if (padding == PaddingType::NONE) {
        valid = false;
        return 0;
    }


    if (padding == PaddingType::ZERO) {
        valid = true;
        return 0;
    }

    if (padding == PaddingType::MIRROR) {
        int mirrored_y = mirror_index(y, img.rows);
        int mirrored_x = mirror_index(x, img.cols);

        valid = true;
        return img.at<cv::Vec3b>(mirrored_y, mirrored_x)[c];
    }

    // Safety fallback.
    valid = false;
    return 0;
}


// ---------------------------------------------------------------------------------------
// Blur

cv::Mat blur_serial(const cv::Mat& input, int radius, PaddingType padding) {
    cv::Mat output(input.rows, input.cols, input.type());

    // get the full fiter size
    int kernel_size = 2 * radius + 1;
    int kernel_area = kernel_size * kernel_size;

    // blurring
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            cv::Vec3b pixel;

            // blurring per a channel
            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                int count = 0; // num of pixels counted
                
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        bool valid = false;
                        unsigned char value = get_pixel_value(
                            input, 
                            y + dy, 
                            x + dx, 
                            c,
                            padding,
                            valid
                        );

                        if (valid) {
                            sum += value;
                            count++;
                        }
                    }
                }

                int divisor = (padding == PaddingType::NONE) ? count : kernel_area;
                if (divisor == 0) {
                    pixel[c] = 0;
                } else {
                    pixel[c] = static_cast<unsigned char>(sum / divisor);
                }
            }

            output.at<cv::Vec3b>(y, x) = pixel;
        }
    }

    return output;
}



cv::Mat blur_openmp(const cv::Mat& input, int radius, PaddingType padding) {
    cv::Mat output(input.rows, input.cols, input.type());

    // get the full fiter size
    int kernel_size = 2 * radius + 1;
    int kernel_area = kernel_size * kernel_size;

  
    // parallel blurring
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            cv::Vec3b pixel;

            // blurring per a channel
            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                int count = 0; // num of pixels counted
                
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        bool valid = false;
                        unsigned char value = get_pixel_value(
                            input, 
                            y + dy, 
                            x + dx, 
                            c,
                            padding,
                            valid
                        );

                        if (valid) {
                            sum += value;
                            count++;
                        }
                    }
                }

                int divisor = (padding == PaddingType::NONE) ? count : kernel_area;
                if (divisor == 0) {
                    pixel[c] = 0;
                } else {
                    pixel[c] = static_cast<unsigned char>(sum / divisor);
                }
            }

            output.at<cv::Vec3b>(y, x) = pixel;
        }
    }

    return output;
}




cv::Mat blur_docker_chunk(const cv::Mat& chunk, int radius, int top_halo_rows, int output_rows, PaddingType padding) {
    // output contains [real rows (not halo) - chunk cols]
    cv::Mat output(output_rows, chunk.cols, chunk.type());

    // get the full kernel size
    int kernel_size = 2 * radius + 1;
    int kernel_area = kernel_size * kernel_size;

    // blurring with nodes
    for (int out_y = 0; out_y < output_rows; ++out_y) {
        int chunk_real_y = top_halo_rows + out_y;

        for (int x = 0; x < chunk.cols; ++x) {
            cv::Vec3b pixel;

            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                int count = 0; // num of pixels counted
                
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        bool valid = false;
                        unsigned char value = get_pixel_value(
                            chunk, 
                            chunk_real_y + dy, 
                            x + dx, 
                            c,
                            padding,
                            valid
                        );

                        if (valid) {
                            sum += value;
                            count++;
                        }
                    }
                }

                int divisor = (padding == PaddingType::NONE) ? count : kernel_area;
                if (divisor == 0) {
                    pixel[c] = 0;
                } else {
                    pixel[c] = static_cast<unsigned char>(sum / divisor);
                }
            }

            output.at<cv::Vec3b>(out_y, x) = pixel;
        }
    }

    return output;
}