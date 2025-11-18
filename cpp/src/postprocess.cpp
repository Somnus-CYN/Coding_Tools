#include "postprocess.h"
#include "utils.h"
#include <algorithm>
#include <cmath>

namespace {
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float area1 = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area2 = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area1 + area2 - inter + 1e-6f);
}

void nms(std::vector<Detection>& dets, float thresh) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> keep;
    for (const auto& d : dets) {
        bool suppress = false;
        for (const auto& k : keep) {
            if (d.class_id == k.class_id && iou(d, k) > thresh) {
                suppress = true;
                break;
            }
        }
        if (!suppress) keep.push_back(d);
    }
    dets.swap(keep);
}
}

std::vector<Detection> decode_and_nms(const ncnn::Mat& out, float conf_thresh, float nms_thresh, float scale, int pad_w, int pad_h, int orig_w, int orig_h) {
    std::vector<Detection> dets;
    // Expect out shape [N, 85]
    for (int i = 0; i < out.h; ++i) {
        const float* row = out.row(i);
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float obj = sigmoid(row[4]);
        const float* cls_ptr = row + 5;

        // Find best class
        int class_id = 0;
        float max_cls = -1e6f;
        for (int c = 0; c < out.w - 5; ++c) {
            float cls_score = sigmoid(cls_ptr[c]);
            if (cls_score > max_cls) {
                max_cls = cls_score;
                class_id = c;
            }
        }
        float score = obj * max_cls;
        if (score < conf_thresh) continue;

        float x1 = (cx - w * 0.5f - pad_w) / scale;
        float y1 = (cy - h * 0.5f - pad_h) / scale;
        float x2 = (cx + w * 0.5f - pad_w) / scale;
        float y2 = (cy + h * 0.5f - pad_h) / scale;

        // Clamp to image size
        x1 = std::max(0.f, std::min(x1, static_cast<float>(orig_w - 1)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(orig_h - 1)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(orig_w - 1)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(orig_h - 1)));

        dets.push_back({x1, y1, x2, y2, score, class_id});
    }

    nms(dets, nms_thresh);
    return dets;
}
