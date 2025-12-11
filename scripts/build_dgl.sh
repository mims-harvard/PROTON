#!/bin/bash
# Build script for DGL from source
# This script builds DGL 2.2.1 from source for platforms where wheels are not available

set -e

DGL_VERSION="v2.2.1"
DGL_REPO="https://github.com/dmlc/dgl.git"
BUILD_DIR="${BUILD_DIR:-/tmp/dgl-build}"
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Building DGL ${DGL_VERSION} from source..."
echo "Python version: ${PYTHON_VERSION}"

# Check if DGL is already installed and is the correct version
if python -c "import dgl; print(dgl.__version__)" 2>/dev/null | grep -q "2.2.1"; then
    echo "DGL 2.2.1 is already installed. Skipping build."
    exit 0
fi

# Check for required build tools
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake is required but not installed."
    echo "Install it with: brew install cmake"
    exit 1
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Clone DGL if not already cloned
if [ ! -d "dgl" ]; then
    echo "Cloning DGL repository..."
    git clone --recursive --branch "${DGL_VERSION}" "${DGL_REPO}" dgl
fi

cd dgl

# Checkout the correct version
git checkout "${DGL_VERSION}"
git submodule update --init --recursive

# Build the C++ library first
echo "Building DGL C++ library..."
mkdir -p build
cd build

# Configure with CMake
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

cd ../python

# Install the Python package
echo "Installing DGL Python package..."
pip install -e . --no-build-isolation --no-deps

echo "DGL ${DGL_VERSION} has been successfully built and installed!"
