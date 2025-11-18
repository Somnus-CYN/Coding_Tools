#include "yolo11n_detector.h"
#include "engine_ncnn.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"
#include <memory>
#include <vector>

class Yolo11nDetector {
public:
    Yolo11nDetector(const std::string& param, const std::string& bin, int input_w, int input_h, float conf, float nms)
        : conf_thresh_(conf), nms_thresh_(nms) {
        engine_ = std::make_unique<EngineNCNN>();
        engine_->set_input_size(input_w, input_h);
        if (!engine_->load(param, bin)) {
            throw std::runtime_error("Failed to load NCNN model");
        }
    }

    std::vector<Detection> infer(const unsigned char* data, int w, int h, int c) {
        ScopedTimer timer("infer");
        auto prep = preprocess_image(data, w, h, c, engine_->input_width(), engine_->input_height());
        if (prep.blob.empty()) return {};

        EngineOutput out;
        if (!engine_->infer(prep.blob, out)) {
            LOGE("Engine inference failed");
            return {};
        }

        // For YOLO11n we expect a single output blob.
        return decode_and_nms(out.blobs[0], conf_thresh_, nms_thresh_, prep.scale, prep.pad_w, prep.pad_h, w, h);
    }

private:
    std::unique_ptr<EngineNCNN> engine_;
    float conf_thresh_;
    float nms_thresh_;
};

// C API wrapper
extern "C" {

struct HandleWrapper {
    std::unique_ptr<Yolo11nDetector> detector;
};

Yolo11nHandle yolo11n_create(const char* param_path, const char* bin_path, int input_width, int input_height, float conf_thresh, float nms_thresh) {
    try {
        auto wrapper = new HandleWrapper();
        wrapper->detector = std::make_unique<Yolo11nDetector>(param_path, bin_path, input_width, input_height, conf_thresh, nms_thresh);
        return reinterpret_cast<Yolo11nHandle>(wrapper);
    } catch (const std::exception& e) {
        LOGE("yolo11n_create failed: %s", e.what());
        return nullptr;
    }
}

void yolo11n_destroy(Yolo11nHandle handle) {
    if (!handle) return;
    auto wrapper = reinterpret_cast<HandleWrapper*>(handle);
    delete wrapper;
}

int yolo11n_detect(Yolo11nHandle handle, const unsigned char* image_data, int img_width, int img_height, int img_channels, Yolo11nBox* out_boxes, int max_boxes) {
    if (!handle || !image_data || !out_boxes || max_boxes <= 0) return -1;
    auto wrapper = reinterpret_cast<HandleWrapper*>(handle);
    auto dets = wrapper->detector->infer(image_data, img_width, img_height, img_channels);
    int count = std::min(static_cast<int>(dets.size()), max_boxes);
    for (int i = 0; i < count; ++i) {
        out_boxes[i].x1 = dets[i].x1;
        out_boxes[i].y1 = dets[i].y1;
        out_boxes[i].x2 = dets[i].x2;
        out_boxes[i].y2 = dets[i].y2;
        out_boxes[i].score = dets[i].score;
        out_boxes[i].class_id = dets[i].class_id;
    }
    return count;
}

const char* yolo11n_version() {
    return "0.1.0-demo";
}

}  // extern "C"
