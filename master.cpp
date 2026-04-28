#include "blur.h"
#include "blur_mpi.h"
#include "image_utils.h"

#include <mpi.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>

namespace fs = std::filesystem;

// ───── Docker chunk info ─────
struct ChunkInfo {
    int id;
    int output_start_row;
    int output_rows;
    int input_start_row;
    int top_halo_rows;
    fs::path input_path;
    fs::path output_path;
    fs::path meta_path;
};

// ───── Padding helpers ─────
static PaddingType parse_padding(const std::string& text) {
    if (text == "zero")   return PaddingType::ZERO;
    if (text == "mirror") return PaddingType::MIRROR;
    if (text == "none")   return PaddingType::NONE;
    throw std::runtime_error("Invalid padding type. Use: zero, mirror, or none");
}

static std::string padding_to_string(PaddingType padding) {
    if (padding == PaddingType::ZERO)   return "zero";
    if (padding == PaddingType::MIRROR) return "mirror";
    if (padding == PaddingType::NONE)   return "none";
    throw std::runtime_error("Unknown padding type");
}

// ───── Shell / Docker helpers ─────
static std::string quote(const std::string& s) {
    std::string result = "\"";
    for (char ch : s) {
        if (ch == '"') result += "\\\"";
        else           result += ch;
    }
    result += "\"";
    return result;
}

static void run_command_or_fail(const std::string& command) {
    std::cout << "\n$ " << command << std::endl;
    int code = std::system(command.c_str());
    if (code != 0) throw std::runtime_error("Command failed: " + command);
}

static std::string docker_command() {
    const char* env = std::getenv("DOCKER_CMD");
    if (env && std::string(env).size() > 0) return std::string(env);
    return "docker";
}

// ───── Metadata I/O ─────
static void write_metadata(const ChunkInfo& info, int radius, PaddingType padding) {
    std::ofstream file(info.meta_path);
    if (!file) throw std::runtime_error("Could not write metadata file: " + info.meta_path.string());

    file << "radius="           << radius << "\n";
    file << "top_halo_rows="    << info.top_halo_rows << "\n";
    file << "output_rows="      << info.output_rows << "\n";
    file << "padding="          << padding_to_string(padding) << "\n";
    file << "chunk_id="         << info.id << "\n";
    file << "output_start_row=" << info.output_start_row << "\n";
    file << "input_start_row="  << info.input_start_row << "\n";
}

// ───── Chunk preparation (Docker) ─────
static std::vector<ChunkInfo> prepare_chunks(
    const cv::Mat& image, int workers, int radius,
    PaddingType padding, const fs::path& chunks_dir
) {
    std::vector<ChunkInfo> chunks;
    fs::create_directories(chunks_dir);

    int rows_per_worker = (image.rows + workers - 1) / workers;

    for (int id = 0; id < workers; ++id) {
        int output_start_row = id * rows_per_worker;
        if (output_start_row >= image.rows) break;

        int output_end_row = std::min(image.rows, output_start_row + rows_per_worker);
        int output_rows    = output_end_row - output_start_row;

        int input_start_row = 0, top_halo_rows = 0;
        cv::Mat chunk = extract_chunk_with_halo(image, output_start_row, output_rows,
                                                 radius, input_start_row, top_halo_rows);

        ChunkInfo info;
        info.id               = id;
        info.output_start_row = output_start_row;
        info.output_rows      = output_rows;
        info.input_start_row  = input_start_row;
        info.top_halo_rows    = top_halo_rows;
        info.input_path  = chunks_dir / ("chunk_" + std::to_string(id) + ".png");
        info.output_path = chunks_dir / ("out_"   + std::to_string(id) + ".png");
        info.meta_path   = chunks_dir / ("meta_"  + std::to_string(id) + ".txt");

        save_image_or_fail(info.input_path.string(), chunk);
        write_metadata(info, radius, padding);
        chunks.push_back(info);
    }
    return chunks;
}

// ───── Merge (Docker) ─────
static cv::Mat merge_chunks(const std::vector<ChunkInfo>& chunks, int full_rows, int full_cols) {
    cv::Mat merged(full_rows, full_cols, CV_8UC3);
    for (const ChunkInfo& info : chunks) {
        cv::Mat part = load_color_image_or_fail(info.output_path.string());
        if (part.rows != info.output_rows || part.cols != full_cols)
            throw std::runtime_error("Unexpected output size from worker " + std::to_string(info.id));

        cv::Rect roi(0, info.output_start_row, full_cols, info.output_rows);
        part.copyTo(merged(roi));
    }
    return merged;
}

// ───── Docker blur ─────
static cv::Mat blur_using_docker(
    const cv::Mat& image, int workers, int radius,
    const fs::path& project_dir, const fs::path& work_dir, PaddingType padding
) {
    fs::path chunks_dir = work_dir / "chunks";
    fs::remove_all(chunks_dir);

    std::vector<ChunkInfo> chunks = prepare_chunks(image, workers, radius, padding, chunks_dir);

    std::string docker = docker_command();
    run_command_or_fail(docker + " build -t cpp-blur-worker " + quote(project_dir.string()));

    fs::path absolute_work_dir = fs::absolute(work_dir);
    std::string name_prefix = "cpp_blur_" +
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    for (const ChunkInfo& info : chunks) {
        std::string container_name = name_prefix + "_" + std::to_string(info.id);
        std::ostringstream cmd;
        cmd << docker << " run -d "
            << "--name " << container_name << " "
            << "-v " << quote(absolute_work_dir.string() + ":/work") << " "
            << "cpp-blur-worker "
            << "/work/chunks/" << info.input_path.filename().string()  << " "
            << "/work/chunks/" << info.output_path.filename().string() << " "
            << "/work/chunks/" << info.meta_path.filename().string();
        run_command_or_fail(cmd.str());
    }

    for (const ChunkInfo& info : chunks) {
        std::string container_name = name_prefix + "_" + std::to_string(info.id);
        run_command_or_fail(docker + " wait " + container_name);
        run_command_or_fail(docker + " logs " + container_name);
        run_command_or_fail(docker + " rm "   + container_name);
    }

    return merge_chunks(chunks, image.rows, image.cols);
}

// ───── Timer wrapper ─────
template <typename Function>
static cv::Mat time_image_function(const std::string& label, Function func, double& milliseconds) {
    auto start  = std::chrono::high_resolution_clock::now();
    cv::Mat result = func();
    auto end    = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << label << " time: " << milliseconds << " ms\n";
    return result;
}

// ───── Comparison log ─────
static void append_comparison_log(
    const fs::path& log_path, int run_id,
    const fs::path& input_path, const fs::path& output_dir,
    int workers, int filter_size, int radius, PaddingType padding,
    int image_cols, int image_rows,
    double serial_ms, double openmp_ms, double docker_ms, double mpi_ms
) {
    std::ofstream log_file(log_path, std::ios::app);
    if (!log_file) throw std::runtime_error("Could not open log file: " + log_path.string());

    double openmp_speedup = (openmp_ms > 0.0) ? (serial_ms / openmp_ms) : 0.0;
    double docker_speedup = (docker_ms > 0.0) ? (serial_ms / docker_ms) : 0.0;
    double mpi_speedup    = (mpi_ms    > 0.0) ? (serial_ms / mpi_ms)    : 0.0;

    log_file << "\n";
    log_file << "==================== Run #" << run_id << " ====================\n";
    log_file << "input_image        => " << input_path.string() << "\n";
    log_file << "output_directory   => " << output_dir.string() << "\n";
    log_file << "image_width        => " << image_cols << "\n";
    log_file << "image_height       => " << image_rows << "\n";
    log_file << "workers            => " << workers << "\n";
    log_file << "filter_size        => " << filter_size << " x " << filter_size << "\n";
    log_file << "radius             => " << radius << "\n";
    log_file << "padding_mode       => " << padding_to_string(padding) << "\n";
    log_file << "serial_time_ms     => " << serial_ms << "\n";
    log_file << "openmp_time_ms     => " << openmp_ms << "\n";
    log_file << "docker_time_ms     => " << docker_ms << "\n";
    log_file << "mpi_time_ms        => " << mpi_ms << "\n";
    log_file << "openmp_speedup     => " << openmp_speedup << "x\n";
    log_file << "docker_speedup     => " << docker_speedup << "x\n";
    log_file << "mpi_speedup        => " << mpi_speedup    << "x\n";
    log_file << "==================== End Run #" << run_id << " ====================\n";
}

// ═══════════════════════════════════════════════════════════
//  main — all MPI ranks enter here
// ═══════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ── Worker ranks ──
    if (rank != 0) {
        mpi_worker_loop();
        MPI_Finalize();
        return 0;
    }

    // ── Master rank (rank 0) ──
    try {
        if (argc != 8) {
            std::cerr << "Usage: master <input_image> <output_directory> <workers>"
                         " <filter_size> <padding_mode> <log_file> <run_id>\n";
            std::cerr << "Example: mpirun -np 5 ./master input.jpg results 4 7 mirror log.txt 1\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        fs::path    input_path  = argv[1];
        fs::path    output_dir  = argv[2];
        int         workers     = std::stoi(argv[3]);
        int         filter_size = std::stoi(argv[4]);
        PaddingType padding     = parse_padding(argv[5]);
        fs::path    log_path    = argv[6];
        int         run_id      = std::stoi(argv[7]);

        if (filter_size < 3 || filter_size % 2 == 0)
            throw std::runtime_error("Kernel size should be odd >= 3");
        if (workers <= 0)
            throw std::runtime_error("workers must be greater than zero");

        int radius = filter_size / 2;

        fs::create_directories(output_dir);
        fs::path project_dir = fs::current_path();
        fs::path work_dir    = output_dir / "docker_work";
        fs::create_directories(work_dir);

        cv::Mat image = load_color_image_or_fail(input_path.string());

        std::cout << "Input image : " << input_path << "\n";
        std::cout << "Image size  : " << image.cols << " x " << image.rows << "\n";
        std::cout << "Workers     : " << workers << "\n";
        std::cout << "MPI procs   : " << size << "  (1 master + " << size - 1 << " workers)\n";
        std::cout << "Radius      : " << radius << "\n";
        std::cout << "Padding     : " << padding_to_string(padding) << "\n";

        double serial_ms = 0.0, openmp_ms = 0.0, docker_ms = 0.0, mpi_ms = 0.0;

        // 1. Serial
        cv::Mat serial_result = time_image_function("Serial", [&]() {
            return blur_serial(image, radius, padding);
        }, serial_ms);
        save_image_or_fail((output_dir / "blur_serial.png").string(), serial_result);

        // 2. OpenMP
        cv::Mat openmp_result = time_image_function("OpenMP", [&]() {
            return blur_openmp(image, radius, padding);
        }, openmp_ms);
        save_image_or_fail((output_dir / "blur_openmp.png").string(), openmp_result);

        // 3. Docker
        cv::Mat docker_result = time_image_function("Docker", [&]() {
            return blur_using_docker(image, workers, radius, project_dir, work_dir, padding);
        }, docker_ms);
        save_image_or_fail((output_dir / "blur_docker.png").string(), docker_result);

        // 4. MPI  ← NEW
        cv::Mat mpi_result = time_image_function("MPI", [&]() {
            return blur_mpi(image, radius, padding);
        }, mpi_ms);
        save_image_or_fail((output_dir / "blur_mpi.png").string(), mpi_result);

        // Summary
        std::cout << "\nSaved outputs:\n";
        std::cout << "  " << (output_dir / "blur_serial.png") << "\n";
        std::cout << "  " << (output_dir / "blur_openmp.png") << "\n";
        std::cout << "  " << (output_dir / "blur_docker.png") << "\n";
        std::cout << "  " << (output_dir / "blur_mpi.png")    << "\n";

        std::cout << "\nTiming comparison:\n";
        std::cout << "  Serial : " << serial_ms << " ms\n";
        std::cout << "  OpenMP : " << openmp_ms << " ms\n";
        std::cout << "  Docker : " << docker_ms << " ms\n";
        std::cout << "  MPI    : " << mpi_ms    << " ms\n";

        append_comparison_log(
            log_path, run_id, input_path, output_dir,
            workers, filter_size, radius, padding,
            image.cols, image.rows,
            serial_ms, openmp_ms, docker_ms, mpi_ms
        );

        std::cout << "\nTiming log saved to: " << log_path.string() << "\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "Error from Main: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    MPI_Finalize();
    return 0;
}