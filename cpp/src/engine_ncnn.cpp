#include "engine_ncnn.h"
#include "utils.h"

EngineNCNN::EngineNCNN() : net_(std::make_unique<ncnn::Net>()) {}
EngineNCNN::~EngineNCNN() = default;

bool EngineNCNN::load(const std::string& param, const std::string& bin) {
    // TODO: adjust options for Vulkan or FP16 as needed.
    net_->opt.num_threads = 4;
    net_->opt.use_vulkan_compute = false;
    if (net_->load_param(param.c_str()) != 0) {
        LOGE("Failed to load param: %s", param.c_str());
        return false;
    }
    if (net_->load_model(bin.c_str()) != 0) {
        LOGE("Failed to load bin: %s", bin.c_str());
        return false;
    }
    return true;
}

bool EngineNCNN::infer(const ncnn::Mat& in, EngineOutput& output) const {
    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(true);
    ex.input("images", in);

    ncnn::Mat out;
    if (ex.extract("output0", out) != 0) {
        LOGE("Failed to extract output0");
        return false;
    }

    output.blobs.clear();
    output.blobs.push_back(out);
    return true;
}
