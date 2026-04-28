#!/usr/bin/env bash
set -e

# -- Default arguments
INPUT_IMAGE="${1:-input_output/input.jpg}"
OUTPUT_BASE_DIR="${2:-input_output/results}"
WORKERS_LIST=(2 4 8)
FILTER_SIZES=(15 31 51)
PADDING_MODES=("zero" "mirror" "none")
LOG_FILE="input_output/log_file.txt"
RUN_ID=1

# -- Validate inputs
if [ -z "$INPUT_IMAGE" ]; then
    echo "Usage: ./run.sh [input_image] [output_base_dir]"
    exit 1
fi

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    exit 1
fi

# -- Docker dependency
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Trying to install docker.io..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable docker >/dev/null 2>&1 || true
    sudo systemctl start docker  >/dev/null 2>&1 || true
    sudo service docker start    >/dev/null 2>&1 || true
fi

if docker ps >/dev/null 2>&1; then
    export DOCKER_CMD="docker"
else
    export DOCKER_CMD="sudo docker"
fi

# -- MPI dependency
if ! command -v mpic++ >/dev/null 2>&1; then
    echo "MPI not found. Installing OpenMPI..."
    sudo apt-get update
    sudo apt-get install -y libopenmpi-dev openmpi-bin
fi

# -- Compile with MPI + OpenMP
echo "================================ Compiling C++ master program (MPI + OpenMP) ===================================="

mpic++ -std=c++17 -O3 -fopenmp \
    master.cpp blur.cpp blur_mpi.cpp image_utils.cpp \
    -o master \
    $(pkg-config --cflags --libs opencv4)

mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
rm -f "$LOG_FILE"

echo "========================================== Experiments ============================================="
echo ">> Input image       : $INPUT_IMAGE"
echo ">> Base output dir   : $OUTPUT_BASE_DIR"
echo ">> Workers tested    : ${WORKERS_LIST[*]}"
echo ">> Filter sizes      : ${FILTER_SIZES[*]}"
echo ">> Padding modes     : ${PADDING_MODES[*]}"

for padding in "${PADDING_MODES[@]}"; do
    for filter_size in "${FILTER_SIZES[@]}"; do
        for workers in "${WORKERS_LIST[@]}"; do
            RUN_OUTPUT_DIR="$OUTPUT_BASE_DIR/${padding}_padding/filter_${filter_size}/workers_${workers}"
            mkdir -p "$RUN_OUTPUT_DIR"

            echo "------------------------------------------------------------"
            echo "Running:"
            echo "  padding     = $padding"
            echo "  filter_size = $filter_size"
            echo "  workers     = $workers"
            echo "  output_dir  = $RUN_OUTPUT_DIR"
            echo "------------------------------------------------------------"

            # Launch with (workers + 1) MPI processes: 1 master + N workers
            MPI_PROCS=$((workers + 1))

            mpirun --allow-run-as-root --oversubscribe -np "$MPI_PROCS" \
                ./master \
                "$INPUT_IMAGE" \
                "$RUN_OUTPUT_DIR" \
                "$workers" \
                "$filter_size" \
                "$padding" \
                "$LOG_FILE" \
                "$RUN_ID"

            RUN_ID=$((RUN_ID + 1))
            echo
        done
    done
done

echo "========================================== The End ============================================="