#!/bin/bash
set -e  # Exit on error

echo "============================================"
echo "Decord Installation Script (CUDA-enabled)"
echo "============================================"

# Step 1: Uninstall any existing decord
echo ""
echo "[1/6] Uninstalling any existing decord..."
pip uninstall decord -y 2>/dev/null || echo "No existing decord installation found"

# Step 2: Install system dependencies
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake
sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev

# Step 3: Clone the repo (if not already cloned)
echo ""
echo "[3/6] Cloning decord repository..."
DECORD_DIR="/home/ubuntu/sky_workdir/decord"
if [ -d "$DECORD_DIR" ]; then
    echo "Decord directory already exists, removing and re-cloning..."
    rm -rf "$DECORD_DIR"
fi
git clone --recursive https://github.com/dmlc/decord "$DECORD_DIR"

# Step 4: Build with CUDA support
echo ""
echo "[4/6] Building decord with CUDA support..."
cd "$DECORD_DIR"
mkdir -p build && cd build

# Try to find CUDA
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "$CUDA_PATH" ]; then
    echo "Found CUDA at: $CUDA_PATH"
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
else
    echo "WARNING: CUDA not found at $CUDA_PATH, building without CUDA..."
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
fi

# Build using all available cores
make -j$(nproc)

# Step 5: Check for libnvcuvid.so issues
echo ""
echo "[5/6] Checking for NVDEC library..."
if ldconfig -p | grep -q libnvcuvid; then
    echo "libnvcuvid.so found"
else
    echo "WARNING: libnvcuvid.so not found. GPU decoding may not work."
    echo "You may need to install NVIDIA Video Codec SDK or check your driver installation."
fi

# Step 6: Install Python bindings
echo ""
echo "[6/6] Installing Python bindings..."
cd "$DECORD_DIR/python"
pip install -e .

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python3 -c "import decord; print(f'Decord version: {decord.__version__}'); from decord import gpu; print('GPU support available')" && echo "SUCCESS: Decord installed with GPU support!" || echo "Decord installed (GPU support may not be available)"

echo ""
echo "Installation complete! Decord installed at: $DECORD_DIR"
