# Integration Guide (Android/iOS)

This document summarizes how to consume the NCNN-based YOLO11n SDK from mobile native code.

## Artifacts
- `models/yolo11n.param`, `models/yolo11n.bin`: NCNN model files.
- `include/yolo11n_detector.h`: Public C API.
- `libs/<platform>/libyolo11n_sdk.a`: Static library built for your target ABI.

## Usage steps
1. Include `yolo11n_detector.h` in your native layer (JNI/ObjC++).
2. Load model files from app assets to an accessible path (e.g., cache dir), then pass absolute paths to `yolo11n_create`.
3. For each frame, convert the image to a contiguous RGB or RGBA buffer and call `yolo11n_detect`.
4. Map `class_id` to labels using the class list in `python/config.yaml`.

## Android build hints (TODO)
- Build `yolo11n_sdk` with Android NDK, linking against NCNN built for the same ABI.
- Add OpenMP/NEON flags as appropriate.
- Ensure `libyolo11n_sdk.a` and NCNN libraries are packaged with your APK/AAR.

## iOS build hints (TODO)
- Build NCNN and `yolo11n_sdk` as static libraries for arm64.
- Link the library in your Xcode project and expose the C API to Swift via a bridging header if needed.

## Desktop validation
- Use `cpp/tests/test_infer_desktop` to validate correctness before shipping to mobile.
