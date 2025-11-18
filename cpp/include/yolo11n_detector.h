// Public C API for the YOLO11n NCNN-based detector.
// This header is what Android/iOS teams should include in their native layers.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the detector instance.
typedef void* Yolo11nHandle;

// Bounding box result in absolute pixel coordinates relative to the original image.
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} Yolo11nBox;

// Container for results. The caller provides an array of Yolo11nBox and its capacity.
typedef struct {
    Yolo11nBox* boxes;  // caller-allocated array
    int count;          // number of valid detections written
} Yolo11nResult;

// Create detector from NCNN param/bin files and desired input resolution (e.g., 640x640).
// Returns nullptr on failure.
Yolo11nHandle yolo11n_create(const char* param_path, const char* bin_path, int input_width, int input_height, float conf_thresh, float nms_thresh);

// Destroy a detector instance.
void yolo11n_destroy(Yolo11nHandle handle);

// Run detection on a raw image buffer (RGB or RGBA).
// image_data: pointer to uint8 image data in row-major order
// img_width/img_height: original resolution
// img_channels: 3 for RGB, 4 for RGBA. Internally converted to RGB.
// out_boxes: caller-provided array with capacity max_boxes
// Returns the number of detections written (>=0) or negative on error.
int yolo11n_detect(Yolo11nHandle handle, const unsigned char* image_data, int img_width, int img_height, int img_channels, Yolo11nBox* out_boxes, int max_boxes);

// Optional helper: get SDK version string.
const char* yolo11n_version();

#ifdef __cplusplus
}
#endif
