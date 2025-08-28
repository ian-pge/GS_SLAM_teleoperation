#!/usr/bin/env bash

echo "Starting dependency installation and build process..."
sudo apt update
sudo apt install -y cmake libgl1-mesa-dev libglew-dev nlohmann-json3-dev
cd /workspace/interactive_viewer/UnityPlugin
mkdir -p build
cmake -B build
cmake --build build
mkdir -p /workspace/interactive_viewer/UnityProject/Assets/GaussianSplattingPlugin/Plugins/
cp /workspace/interactive_viewer/UnityPlugin/build/libgaussiansplatting.so /workspace/interactive_viewer/UnityProject/Assets/GaussianSplattingPlugin/Plugins
