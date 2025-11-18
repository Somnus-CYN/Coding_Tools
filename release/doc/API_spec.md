# YOLO11n NCNN SDK C API

## Handle lifecycle
- `Yolo11nHandle yolo11n_create(const char* param_path, const char* bin_path, int input_width, int input_height, float conf_thresh, float nms_thresh);`
  - Loads NCNN `.param` and `.bin` files and sets preprocessing size.
- `void yolo11n_destroy(Yolo11nHandle handle);`

## Inference
- `int yolo11n_detect(Yolo11nHandle handle, const unsigned char* image_data, int img_width, int img_height, int img_channels, Yolo11nBox* out_boxes, int max_boxes);`
  - `image_data`: contiguous RGB or RGBA buffer.
  - `img_channels`: 3 or 4.
  - `out_boxes`: caller-allocated array.
  - Returns number of detections or negative on error.

## Data structures
```c
typedef struct {
    float x1, y1, x2, y2; // pixel coordinates in original image space
    float score;          // confidence score after NMS
    int class_id;         // zero-based class index
} Yolo11nBox;
```

## Version
- `const char* yolo11n_version();` returns the SDK version string.

## Threading
- The current implementation uses NCNN default threading (4 threads). Adjust in `engine_ncnn.cpp` if needed.
