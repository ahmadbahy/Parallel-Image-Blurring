# Use Ubuntu as the base Linux system for the worker container.
FROM ubuntu:22.04

# Avoid interactive installation questions while building the Docker image.
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install the C++ compiler, OpenMP support, pkg-config, and OpenCV.
RUN apt-get update && apt-get install -y \
    g++ \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a folder inside the container for the worker source code.
WORKDIR /app

# Copy the shared blur declarations into the container.
COPY blur.h /app/blur.h

# Copy the shared blur implementation into the container.
COPY blur.cpp /app/blur.cpp

# Copy the image utility declarations into the container.
COPY image_utils.h /app/image_utils.h

# Copy the image utility implementation into the container.
COPY image_utils.cpp /app/image_utils.cpp

# Copy the worker program into the container.
COPY worker.cpp /app/worker.cpp

# Compile the worker program with optimization and OpenCV libraries.
RUN g++ -std=c++17 -O3 worker.cpp blur.cpp image_utils.cpp -o worker $(pkg-config --cflags --libs opencv4)

# The default program run by the container is the worker executable.
ENTRYPOINT ["/app/worker"]
