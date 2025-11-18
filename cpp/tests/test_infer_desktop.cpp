// Simple desktop test using the C API.
// Build: mkdir build && cd build && cmake .. && make -j
// Run: ./yolo11n_test_desktop ../python/config.yaml ../test_data/images
#include "yolo11n_detector.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

struct ConfigPaths {
    std::string param;
    std::string bin;
    std::string test_images;
    int input_size;
    float conf;
    float nms;
};

ConfigPaths load_cfg(const std::string& path) {
    YAML::Node node = YAML::LoadFile(path);
    ConfigPaths cfg;
    cfg.param = node["paths"]["ncnn_param"].as<std::string>();
    cfg.bin = node["paths"]["ncnn_bin"].as<std::string>();
    cfg.test_images = node["paths"]["test_images"].as<std::string>();
    cfg.input_size = node["input_size"].as<int>();
    cfg.conf = node["confidence_threshold"].as<float>();
    cfg.nms = node["nms_iou_threshold"].as<float>();
    return cfg;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> [image_dir]\n";
        return 1;
    }

    ConfigPaths cfg = load_cfg(argv[1]);
    if (argc >= 3) {
        cfg.test_images = argv[2];
    }

    Yolo11nHandle handle = yolo11n_create(cfg.param.c_str(), cfg.bin.c_str(), cfg.input_size, cfg.input_size, cfg.conf, cfg.nms);
    if (!handle) {
        std::cerr << "Failed to create detector" << std::endl;
        return 1;
    }

    nlohmann::json results = nlohmann::json::array();
    for (const auto& entry : std::filesystem::directory_iterator(cfg.test_images)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        std::vector<unsigned char> buffer(img.data, img.data + img.total() * img.channels());
        std::vector<Yolo11nBox> boxes(200);
        int count = yolo11n_detect(handle, buffer.data(), img.cols, img.rows, img.channels(), boxes.data(), boxes.size());
        std::cout << entry.path().filename().string() << ": detections = " << count << std::endl;

        nlohmann::json dets = nlohmann::json::array();
        for (int i = 0; i < count; ++i) {
            dets.push_back({
                {"bbox", {boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2}},
                {"score", boxes[i].score},
                {"class_id", boxes[i].class_id},
            });
        }
        results.push_back({{"image", entry.path().filename().string()}, {"detections", dets}});
    }

    std::ofstream ofs("../release/test_data/desktop_results.json");
    ofs << results.dump(2);

    yolo11n_destroy(handle);
    return 0;
}
