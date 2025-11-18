// Image preprocessing utilities for YOLO11n (resize + letterbox + normalization).
#pragma once
#include <ncnn/mat.h>
#include <cstdint>
#include <tuple>

struct PreprocessResult {
    ncnn::Mat blob;
    float scale;
    int pad_w;
    int pad_h;
};

// Convert raw RGB/RGBA buffer to NCNN input tensor with letterbox resize.
PreprocessResult preprocess_image(const unsigned char* data, int w, int h, int c, int target_w, int target_h);
