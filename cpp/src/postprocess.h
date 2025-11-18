// Postprocessing for YOLO11n outputs (decode + confidence filter + NMS).
#pragma once
#include <ncnn/mat.h>
#include <vector>

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

std::vector<Detection> decode_and_nms(const ncnn::Mat& out, float conf_thresh, float nms_thresh, float scale, int pad_w, int pad_h, int orig_w, int orig_h);
