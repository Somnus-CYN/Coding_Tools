// Lightweight wrapper around the NCNN network for YOLO11n inference.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include <ncnn/net.h>
#include <ncnn/mat.h>

struct EngineOutput {
    std::vector<ncnn::Mat> blobs;
};

class EngineNCNN {
public:
    EngineNCNN();
    ~EngineNCNN();

    bool load(const std::string& param, const std::string& bin);
    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }

    // Run forward pass given preprocessed input.
    bool infer(const ncnn::Mat& in, EngineOutput& output) const;

    void set_input_size(int w, int h) {
        input_w_ = w;
        input_h_ = h;
    }

private:
    std::unique_ptr<ncnn::Net> net_;
    int input_w_ = 640;
    int input_h_ = 640;
};
