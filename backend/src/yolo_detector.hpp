#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>

struct DetectionBox {
    cv::Rect2f box; // [x, y, w, h] in original image space
    float confidence;
    int class_id;
    std::string label;
};

class YoloDetector {
public:
    YoloDetector() = default;
    ~YoloDetector() = default;

    bool init(const std::string& model_path, float conf_thresh = 0.35f, float nms_thresh = 0.45f, bool use_coreml = true, int target_res = 640);
    std::vector<DetectionBox> detect(const cv::Mat& frame);
    std::vector<DetectionBox> detect_batch_simd(const std::vector<cv::Mat>& frames);

    void set_conf_threshold(float val) { conf_threshold_ = val; }
    void set_nms_threshold(float val) { nms_threshold_ = val; }
    float get_conf_threshold() const { return conf_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }

    static const std::vector<std::string> COCO_CLASSES;

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YoloDetector"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    float conf_threshold_ = 0.35f;
    float nms_threshold_ = 0.45f;
    int input_width_ = 640;
    int input_height_ = 640;

    void preprocess(const cv::Mat& frame, std::vector<float>& input_tensor_values, float& rx, float& ry, float& pad_w, float& pad_h);
};

#endif // YOLO_DETECTOR_HPP
