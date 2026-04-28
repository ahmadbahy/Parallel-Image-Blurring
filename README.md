```markdown
# 🖼️ Image Blurring — Parallel Computing Comparison

A C++ project that applies **box blur (mean filter)** to an image using **four parallelization strategies** and benchmarks their performance side by side.

| Method     | Parallelism Model          | Communication      |
|------------|----------------------------|---------------------|
| **Serial** | None (single thread)       | —                   |
| **OpenMP** | Shared-memory (threads)    | Shared variables    |
| **Docker** | Process-level (containers) | Filesystem (files)  |
| **MPI**    | Distributed-memory         | Message passing     |

---

## 📁 Project Structure

```
.
├── master.cpp          # Main orchestrator (MPI rank 0)
├── worker.cpp          # Docker container worker
├── blur.h              # Blur function declarations
├── blur.cpp            # Blur implementations (serial, OpenMP, chunk)
├── blur_mpi.h          # MPI blur declarations
├── blur_mpi.cpp        # MPI master/worker communication logic
├── image_utils.h       # Image I/O and chunking declarations
├── image_utils.cpp     # Image loading, saving, and chunk extraction
├── Dockerfile          # Docker worker image definition
├── run.sh              # Automated experiment runner
├── input_output/       # Default input/output directory
│   ├── input.jpg       # Input image (user-provided)
│   ├── results/        # Generated output images and logs
│   └── log_file.txt    # Timing comparison log
├── .gitignore
└── README.md
```

---

## 🔧 Prerequisites

| Dependency   | Purpose                        | Install (Ubuntu/Debian)                        |
|--------------|--------------------------------|------------------------------------------------|
| **g++**      | C++17 compiler                 | `sudo apt install g++`                         |
| **OpenCV**   | Image I/O and processing       | `sudo apt install libopencv-dev pkg-config`    |
| **OpenMP**   | Thread-level parallelism       | Included with g++ (`-fopenmp`)                 |
| **OpenMPI**  | MPI message passing            | `sudo apt install libopenmpi-dev openmpi-bin`  |
| **Docker**   | Container-based parallelism    | `sudo apt install docker.io`                   |

> **Note:** `run.sh` will attempt to install missing MPI and Docker packages automatically.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Image_Blurring_CPP_Docker.git
cd Image_Blurring_CPP_Docker
```

### 2. Add an Input Image

Place your image in the `input_output/` directory:

```bash
cp /path/to/your/image.jpg input_output/input.jpg
```

### 3. Run All Experiments

```bash
chmod +x run.sh
./run.sh
```

This will:
- Compile the project with `mpic++` (MPI + OpenMP)
- Run **27 experiments** (3 padding modes × 3 filter sizes × 3 worker counts)
- Save all output images and timing logs

### 4. Custom Single Run

```bash
# Compile manually
mpic++ -std=c++17 -O3 -fopenmp \
    master.cpp blur.cpp blur_mpi.cpp image_utils.cpp \
    -o master \
    $(pkg-config --cflags --libs opencv4)

# Run with 4 MPI workers (5 processes = 1 master + 4 workers)
mpirun --oversubscribe -np 5 ./master \
    input_output/input.jpg \
    input_output/results \
    4 \
    7 \
    mirror \
    input_output/log_file.txt \
    1
```

---

## ⚙️ Command-Line Arguments

```
mpirun -np <N+1> ./master <input_image> <output_dir> <workers> <filter_size> <padding> <log_file> <run_id>
```

| Argument        | Description                                  | Example          |
|-----------------|----------------------------------------------|------------------|
| `input_image`   | Path to the input image                      | `input.jpg`      |
| `output_dir`    | Directory for output images                  | `results/`       |
| `workers`       | Number of parallel workers (Docker & MPI)    | `4`              |
| `filter_size`   | Blur kernel size (odd, ≥ 3)                  | `7`              |
| `padding`       | Padding mode: `zero`, `mirror`, or `none`    | `mirror`         |
| `log_file`      | Path to the timing log file                  | `log.txt`        |
| `run_id`        | Experiment identifier for the log            | `1`              |

> **Important:** MPI process count (`-np`) should be `workers + 1` (1 master + N workers).

---

## 🧩 Padding Modes

When the blur kernel extends beyond image boundaries:

| Mode       | Behavior                                          | Visual                     |
|------------|---------------------------------------------------|----------------------------|
| **`zero`** | Outside pixels are treated as black (0)           | Darkened edges             |
| **`mirror`** | Image is reflected at boundaries                | Seamless edges             |
| **`none`** | Only valid pixels are averaged (adaptive divisor) | Slightly brighter edges    |

```
Mirror example (1D, values = [1, 2, 3]):

  ... 2  1 │ 1  2  3 │ 3  2 ...
     mirror │  image  │ mirror
```

---

## 🔄 How Each Method Works

### Serial
Simple nested loop — every pixel is processed sequentially on a single CPU core.

### OpenMP
Same logic as serial, but the outer row loop is parallelized across threads:
```cpp
#pragma omp parallel for schedule(static)
for (int y = 0; y < rows; ++y) { ... }
```

### Docker
1. Image is split into **chunks with halo rows**
2. Each chunk + metadata is saved to disk
3. A Docker container is launched per chunk
4. Containers process their chunks independently
5. Results are read from disk and merged

### MPI
1. Rank 0 splits the image into chunks with halo rows
2. Chunks are sent to worker ranks via `MPI_Send`
3. Workers blur their chunk and send results back via `MPI_Recv`
4. Rank 0 merges all results into the final image

```
  Rank 0 (Master)          Rank 1..N (Workers)
  ┌──────────────┐         ┌──────────────┐
  │ Split image  │────────▶│ Receive chunk│
  │              │ MPI_Send│ Blur chunk   │
  │ Merge results│◀────────│ Send result  │
  └──────────────┘ MPI_Recv└──────────────┘
```

---

## 📊 Output

### Images
Each run produces four blurred images in the output directory:

```
output_dir/
├── blur_serial.png
├── blur_openmp.png
├── blur_docker.png
└── blur_mpi.png
```

### Timing Log (`log_file.txt`)

```
==================== Run #1 ====================
input_image        => input_output/input.jpg
image_width        => 1920
image_height       => 1080
workers            => 4
filter_size        => 15 x 15
padding_mode       => mirror
serial_time_ms     => 2450.32
openmp_time_ms     => 620.15
docker_time_ms     => 5200.88
mpi_time_ms        => 580.42
openmp_speedup     => 3.95x
docker_speedup     => 0.47x
mpi_speedup        => 4.22x
==================== End Run #1 ====================
```

---

## 📈 Expected Performance Characteristics

| Method   | Overhead         | Best For                              |
|----------|------------------|---------------------------------------|
| Serial   | None             | Small images, baseline reference      |
| OpenMP   | Very low         | Multi-core single machines            |
| MPI      | Low              | Multi-process / distributed systems   |
| Docker   | High (I/O, boot) | Isolation, heterogeneous environments |

> **Docker is typically slower** than serial for small/medium images due to container startup and filesystem I/O overhead. It demonstrates **process isolation**, not raw speed.

---

## 🧪 Experiment Grid (`run.sh`)

| Parameter       | Values Tested        |
|-----------------|----------------------|
| Padding modes   | `zero`, `mirror`, `none` |
| Filter sizes    | `15×15`, `31×31`, `51×51` |
| Worker counts   | `2`, `4`, `8`            |

**Total: 27 experiments**, each running all 4 blur methods.

---

## 🐳 Docker Notes

- The Docker image is built automatically from the included `Dockerfile`
- Workers run on Ubuntu 22.04 with OpenCV
- The host `output_dir/docker_work/` directory is mounted as `/work` inside containers
- Set `DOCKER_CMD` environment variable if you need `sudo`:
  ```bash
  export DOCKER_CMD="sudo docker"
  ```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Permission denied: ./run.sh` | `chmod +x run.sh` |
| `docker: permission denied` | Add user to docker group: `sudo usermod -aG docker $USER` then re-login |
| `mpirun not found` | `sudo apt install libopenmpi-dev openmpi-bin` |
| `pkg-config opencv4 not found` | `sudo apt install libopencv-dev pkg-config` |
| MPI oversubscribe error | Already handled with `--oversubscribe` flag in `run.sh` |

---

## 📝 License

This project is for academic / educational purposes.
```