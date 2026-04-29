#include "blur_mpi.h"
#include "image_utils.h"

#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <vector>

// MPI message tags
static const int TAG_META   = 1;
static const int TAG_DATA   = 2;
static const int TAG_RESULT = 3;
static const int TAG_DONE   = 4;

// master logic
cv::Mat blur_mpi(const cv::Mat& image, int radius, PaddingType padding) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_workers = size - 1;

    // 1 proc --> serial
    if (num_workers < 1) {
        std::cerr << "[MPI] Only 1 process — falling back to serial blur.\n";
        return blur_serial(image, radius, padding);
    }

    // prepare chunks
    struct ChunkMeta {
        int output_start_row;
        int output_rows;
        int top_halo_rows;
        int chunk_rows;
        int chunk_cols;
    };

    int rows_per_worker = (image.rows + num_workers - 1) / num_workers;

    std::vector<ChunkMeta> metas;
    std::vector<cv::Mat>   chunks;

    for (int i = 0; i < num_workers; ++i) {
        int output_start = i * rows_per_worker;
        if (output_start >= image.rows) break;

        int output_rows = std::min(rows_per_worker, image.rows - output_start);

        int input_start_row = 0;
        int top_halo_rows   = 0;
        cv::Mat chunk = extract_chunk_with_halo(
            image, output_start, output_rows, radius,
            input_start_row, top_halo_rows
        );

        ChunkMeta m;
        m.output_start_row = output_start;
        m.output_rows      = output_rows;
        m.top_halo_rows    = top_halo_rows;
        m.chunk_rows       = chunk.rows;
        m.chunk_cols       = chunk.cols;

        metas.push_back(m);
        chunks.push_back(chunk.isContinuous() ? chunk : chunk.clone());
    }

    int actual_workers = static_cast<int>(chunks.size());

    // distribute work
    for (int i = 0; i < actual_workers; ++i) {
        int dest = i + 1;

        
        int meta_buf[6] = {
            radius,
            static_cast<int>(padding),
            metas[i].top_halo_rows,
            metas[i].output_rows,
            metas[i].chunk_rows,
            metas[i].chunk_cols
        };
        MPI_Send(meta_buf, 6, MPI_INT, dest, TAG_META, MPI_COMM_WORLD);

        
        int data_size = metas[i].chunk_rows * metas[i].chunk_cols * 3;
        MPI_Send(chunks[i].data, data_size, MPI_UNSIGNED_CHAR,
                 dest, TAG_DATA, MPI_COMM_WORLD);
    }

    // no work -> stop Idle
    for (int i = actual_workers; i < num_workers; ++i) {
        int dummy[6] = {0};
        MPI_Send(dummy, 6, MPI_INT, i + 1, TAG_DONE, MPI_COMM_WORLD);
    }

    // merging results
    cv::Mat merged(image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < actual_workers; ++i) {
        int src         = i + 1;
        int result_size = metas[i].output_rows * metas[i].chunk_cols * 3;

        cv::Mat result(metas[i].output_rows, metas[i].chunk_cols, CV_8UC3);
        MPI_Recv(result.data, result_size, MPI_UNSIGNED_CHAR,
                 src, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cv::Rect roi(0, metas[i].output_start_row,
                     metas[i].chunk_cols, metas[i].output_rows);
        result.copyTo(merged(roi));
    }

    return merged;
}

// worker logic
void mpi_worker_loop() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // get metadata
    int meta_buf[6];
    MPI_Status status;
    MPI_Recv(meta_buf, 6, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == TAG_DONE) {
        return;
    }

    int radius        = meta_buf[0];
    PaddingType padding = static_cast<PaddingType>(meta_buf[1]);
    int top_halo_rows = meta_buf[2];
    int output_rows   = meta_buf[3];
    int chunk_rows    = meta_buf[4];
    int chunk_cols    = meta_buf[5];

    // get chunk data
    int data_size = chunk_rows * chunk_cols * 3;
    cv::Mat chunk(chunk_rows, chunk_cols, CV_8UC3);
    MPI_Recv(chunk.data, data_size, MPI_UNSIGNED_CHAR,
             0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // blur
    cv::Mat result = blur_docker_chunk(chunk, radius, top_halo_rows, output_rows, padding);

    // send to master
    int result_size = result.rows * result.cols * 3;
    MPI_Send(result.data, result_size, MPI_UNSIGNED_CHAR,
             0, TAG_RESULT, MPI_COMM_WORLD);

    std::cout << "[MPI Worker " << rank << "] Finished processing "
              << output_rows << " rows.\n";
}