#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <thread>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

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

    void set_conf_threshold(float conf_thresh) { conf_threshold_ = conf_thresh; }
    void set_nms_threshold(float nms_thresh) { nms_threshold_ = nms_thresh; }

private:
    std::vector<DetectionBox> postprocess(const std::vector<float>& output_tensor_values, const std::vector<int64_t>& output_shape, int orig_w, int orig_h);
    void nms(std::vector<DetectionBox>& boxes, float nms_thresh);
    float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YoloDetector"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::string output_name_;

    float conf_threshold_ = 0.35f;
    float nms_threshold_ = 0.45f;
    int input_width_ = 640;
    int input_height_ = 640;

    std::vector<std::string> class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
};

#endif // YOLO_DETECTOR_HPP
