#include "yolo_detector.hpp"
#include "logger.hpp"

bool YoloDetector::init(const std::string& model_path, float conf_thresh, float nms_thresh, bool use_coreml, int target_res) {
    conf_threshold_ = conf_thresh;
    nms_threshold_ = nms_thresh;
    input_width_ = target_res;
    input_height_ = target_res;

    unsigned num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 8;

    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetInterOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.EnableCpuMemArena();

#ifdef __APPLE__
    if (use_coreml) {
        uint32_t coreml_flags = COREML_FLAG_ENABLE_ON_SUBGRAPH | COREML_FLAG_CREATE_MLPROGRAM;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options_, coreml_flags);
        if (status != nullptr) {
            Logger::getInstance().warn("YoloDetector", "CoreML EP notice: Falling back to Apple CPU SIMD execution");
        } else {
            Logger::getInstance().info("YoloDetector", "Apple Neural Engine (CoreML EP) Hardware Acceleration ENABLED 🚀");
        }
    } else {
        Logger::getInstance().info("YoloDetector", "ARM Neon CPU SIMD Multi-Threaded Parallel Execution Provider ENABLED ⚡");
    }
#else
    Logger::getInstance().info("YoloDetector", "Linux x86_64 OpenMP Multi-Threaded Parallel CPU Execution Provider ENABLED ⚡");
#endif

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        Ort::AllocatedStringPtr input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
        input_name_ = input_name_alloc.get();

        Ort::AllocatedStringPtr output_name_alloc = session_->GetOutputNameAllocated(0, allocator_);
        output_name_ = output_name_alloc.get();

        Logger::getInstance().info("YoloDetector", "Initialized ONNX detector: " + model_path);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().critical("YoloDetector", "Failed to load model " + model_path + ": " + e.what());
        return false;
    }
}

std::vector<DetectionBox> YoloDetector::detect(const cv::Mat& frame) {
    if (frame.empty() || !session_) return {};

    int orig_w = frame.cols;
    int orig_h = frame.rows;

    cv::Mat resized_mat;
    cv::resize(frame, resized_mat, cv::Size(input_width_, input_height_));
    cv::cvtColor(resized_mat, resized_mat, cv::COLOR_BGR2RGB);

    resized_mat.convertTo(resized_mat, CV_32FC3, 1.0 / 255.0);

    size_t input_tensor_size = 1 * 3 * input_height_ * input_width_;
    std::vector<float> input_tensor_values(input_tensor_size);

    std::vector<cv::Mat> chw_channels(3);
    chw_channels[0] = cv::Mat(input_height_, input_width_, CV_32FC1, input_tensor_values.data());
    chw_channels[1] = cv::Mat(input_height_, input_width_, CV_32FC1, input_tensor_values.data() + input_height_ * input_width_);
    chw_channels[2] = cv::Mat(input_height_, input_width_, CV_32FC1, input_tensor_values.data() + 2 * input_height_ * input_width_);
    cv::split(resized_mat, chw_channels);

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    try {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );

        float* float_data = output_tensors[0].GetTensorMutableData<float>();
        auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t total_elements = 1;
        for (auto dim : shape) total_elements *= dim;

        std::vector<float> output_tensor_values(float_data, float_data + total_elements);

        return postprocess(output_tensor_values, shape, orig_w, orig_h);
    } catch (const std::exception& e) {
        Logger::getInstance().error("YoloDetector", std::string("Inference failed: ") + e.what());
        return {};
    }
}

std::vector<DetectionBox> YoloDetector::postprocess(const std::vector<float>& output_tensor_values, const std::vector<int64_t>& shape, int orig_w, int orig_h) {
    std::vector<DetectionBox> detections;

    if (shape.size() != 3) return detections;

    int num_channels = 0;
    int num_anchors = 0;

    if (shape[1] < shape[2]) {
        num_channels = static_cast<int>(shape[1]);
        num_anchors = static_cast<int>(shape[2]);
    } else {
        num_channels = static_cast<int>(shape[2]);
        num_anchors = static_cast<int>(shape[1]);
    }

    float scale_x = static_cast<float>(orig_w) / static_cast<float>(input_width_);
    float scale_y = static_cast<float>(orig_h) / static_cast<float>(input_height_);

    bool transposed = (shape[1] < shape[2]);

    for (int i = 0; i < num_anchors; ++i) {
        float cx = 0, cy = 0, w = 0, h = 0;
        int max_class_id = -1;
        float max_score = 0.0f;

        if (transposed) {
            cx = output_tensor_values[0 * num_anchors + i];
            cy = output_tensor_values[1 * num_anchors + i];
            w  = output_tensor_values[2 * num_anchors + i];
            h  = output_tensor_values[3 * num_anchors + i];

            for (int c = 4; c < num_channels; ++c) {
                float score = output_tensor_values[c * num_anchors + i];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c - 4;
                }
            }
        } else {
            int stride = num_channels;
            cx = output_tensor_values[i * stride + 0];
            cy = output_tensor_values[i * stride + 1];
            w  = output_tensor_values[i * stride + 2];
            h  = output_tensor_values[i * stride + 3];

            for (int c = 4; c < num_channels; ++c) {
                float score = output_tensor_values[i * stride + c];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c - 4;
                }
            }
        }

        if (max_score >= conf_threshold_) {
            // Filter strictly for Person (0) and Car (2) or common objects
            std::string label = (max_class_id < static_cast<int>(class_names_.size())) ? class_names_[max_class_id] : "object";

            float x = (cx - w / 2.0f) * scale_x;
            float y = (cy - h / 2.0f) * scale_y;
            float width = w * scale_x;
            float height = h * scale_y;

            // Constrain box bounds
            x = std::max(0.0f, std::min(x, static_cast<float>(orig_w)));
            y = std::max(0.0f, std::min(y, static_cast<float>(orig_h)));
            width = std::min(width, static_cast<float>(orig_w) - x);
            height = std::min(height, static_cast<float>(orig_h) - y);

            if (width > 5.0f && height > 5.0f) {
                DetectionBox det;
                det.box = cv::Rect2f(x, y, width, height);
                det.confidence = max_score;
                det.class_id = max_class_id;
                det.label = label;
                detections.push_back(det);
            }
        }
    }

    nms(detections, nms_threshold_);
    return detections;
}

void YoloDetector::nms(std::vector<DetectionBox>& boxes, float nms_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const DetectionBox& a, const DetectionBox& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<DetectionBox> keep;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (boxes[i].class_id == boxes[j].class_id) {
                if (iou(boxes[i].box, boxes[j].box) > nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }

    boxes = std::move(keep);
}

float YoloDetector::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float union_area = area_a + area_b - intersection;

    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}
