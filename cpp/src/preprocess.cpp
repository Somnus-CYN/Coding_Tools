#include "preprocess.h"
#include "utils.h"
#include <algorithm>
#include <vector>

namespace {
// Helper to convert RGBA to RGB
void rgba_to_rgb(const unsigned char* src, int w, int h, int stride, std::vector<unsigned char>& dst) {
    dst.resize(w * h * 3);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const unsigned char* p = src + i * stride + j * 4;
            unsigned char* q = dst.data() + (i * w + j) * 3;
            q[0] = p[0];
            q[1] = p[1];
            q[2] = p[2];
        }
    }
}
}

PreprocessResult preprocess_image(const unsigned char* data, int w, int h, int c, int target_w, int target_h) {
    const int channels = 3;
    std::vector<unsigned char> rgb;
    const unsigned char* src_ptr = data;

    if (c == 4) {
        rgba_to_rgb(data, w, h, w * 4, rgb);
        src_ptr = rgb.data();
    } else if (c != 3) {
        LOGE("Unsupported channel count: %d", c);
        return {};
    }

    float scale = std::min(target_w * 1.0f / w, target_h * 1.0f / h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    int pad_w = (target_w - new_w) / 2;
    int pad_h = (target_h - new_h) / 2;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(src_ptr, ncnn::Mat::PIXEL_RGB, w, h, new_w, new_h);
    ncnn::Mat out(target_w, target_h, channels);
    out.fill(114.f);
    in.copy_to(out.row_range(pad_h, new_h).col_range(pad_w, new_w));

    // Normalize to 0-1 and convert to planar format for NCNN.
    const float norm = 1.0f / 255.0f;
    for (int q = 0; q < channels; ++q) {
        float* out_ptr = out.channel(q);
        for (int i = 0; i < out.h; ++i) {
            for (int j = 0; j < out.w; ++j) {
                out_ptr[i * out.w + j] *= norm;
            }
        }
    }

    return {out, scale, pad_w, pad_h};
}
