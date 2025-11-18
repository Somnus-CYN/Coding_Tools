# YOLO11n Mobile Demo Pipeline

This repository provides a minimal end-to-end workflow for deploying Ultralytics YOLO11n to mobile devices using NCNN. It includes Python scripts for export and reference inference, and a C++ NCNN-based SDK with a simple C API for Android/iOS integration.

## Layout
- `python/`: Ultralytics + ONNX utilities and comparison scripts.
- `tools/`: ONNX -> NCNN conversion helper.
- `cpp/`: NCNN-backed detector implementation, C API, and desktop test.
- `release_template/`: Expected release packaging structure.
- `release/`: Staging area to collect artifacts (models, headers, libs, docs).
- `test_data/`: Add sample images here for testing.

## Quick start
1. Install Python deps: `pip install -r python/requirements.txt`.
2. Export ONNX: `cd python && python export_onnx.py` (produces `models/yolo11n.onnx`).
3. Generate reference outputs:
   - `python run_ultralytics_ref.py`
   - `python run_onnx_ref.py`
   - `python compare_outputs.py`
4. Convert ONNX -> NCNN: `../tools/convert_onnx_to_ncnn.sh config.yaml`.
5. Build C++ SDK:
   ```
   cd cpp
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ```
6. Run desktop test: `./build/yolo11n_test_desktop ../python/config.yaml ../test_data/images`.

After verification, copy artifacts into `release/` matching `release_template/` and hand off to mobile teams.
