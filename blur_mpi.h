#ifndef BLUR_MPI_H
#define BLUR_MPI_H

#include <opencv2/opencv.hpp>
#include "blur.h"

// Called by rank 0 — distributes chunks to workers and merges results.
cv::Mat blur_mpi(const cv::Mat& image, int radius, PaddingType padding);

// Called by ranks > 0 — waits for work, processes, sends back result.
void mpi_worker_loop();

#endif