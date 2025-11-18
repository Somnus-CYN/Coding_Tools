# Project layout snapshot

This document lists the key files delivered in the YOLO11n-to-NCNN mobile demo skeleton. It mirrors the expected tree so teams can quickly locate scripts, libraries, and docs.

```
.
├─ README.md                    # Quick start overview
├─ PROJECT_TREE.md              # (this file) layout snapshot
├─ python/
│  ├─ requirements.txt          # Ultralytics + ONNX deps
│  ├─ config.yaml               # Model/input paths and thresholds
│  ├─ export_onnx.py            # Export YOLO11n -> ONNX
│  ├─ run_ultralytics_ref.py    # Ultralytics reference detections
│  ├─ run_onnx_ref.py           # ONNX Runtime reference detections
│  └─ compare_outputs.py        # Sanity check of Ultralytics vs ONNX outputs
├─ tools/
│  └─ convert_onnx_to_ncnn.sh   # onnx2ncnn + ncnnoptimize wrapper
├─ cpp/
│  ├─ CMakeLists.txt            # NCNN SDK build + desktop test
│  ├─ include/
│  │  └─ yolo11n_detector.h     # Public C API header
│  ├─ src/
│  │  ├─ engine_ncnn.*          # NCNN network wrapper
│  │  ├─ preprocess.*           # Resize/letterbox/normalize
│  │  ├─ postprocess.*          # Decode + NMS
│  │  ├─ utils.*                # Logging/timing helpers
│  │  └─ yolo11n_detector.cpp   # Detector class + C API glue
│  └─ tests/
│     └─ test_infer_desktop.cpp # Desktop harness calling C API
├─ release_template/            # Target delivery layout
├─ release/                     # Staging area for built artifacts
├─ test_data/                   # Place sample images + reference JSONs
│  └─ results_ref.json          # Placeholder
└─ MarkDown/
   └─ markdown基本语法.ipynb   # Markdown tips (not part of pipeline)
```

Notes:
- Scripts expect `python/config.yaml` for paths; adjust when relocating models.
- The NCNN build assumes headers/libs are discoverable via standard locations—see TODOs in `cpp/CMakeLists.txt`.
- Mobile teams consume `release/include/`, `release/models/`, and `release/libs/` along with `release/doc/`.
