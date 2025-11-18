# YOLO11n Mobile SDK Release Template

This folder sketches the expected release layout for mobile integration. After running the Python export and C++ build steps, copy artifacts into a `release/` directory structured like:

```
release/
  models/
    yolo11n.param
    yolo11n.bin
  include/
    yolo11n_detector.h
  libs/
    linux_x86_64/
      libyolo11n_sdk.a
    android/   # TODO: place NDK-built libs
    ios/       # TODO: place iOS-built libs
  doc/
    API_spec.md
    integration_guide.md
  test_data/
    images/
    ref_ultralytics.json
    ref_onnx.json
```

The mobile teams only need `include/`, `libs/`, and `models/` plus the documentation. Desktop test outputs can go into `test_data/` for reference.
