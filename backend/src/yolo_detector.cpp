#include "yolo_detector.hpp"
#include "logger.hpp"
#include <thread>

const std::vector<std::string> YoloDetector::COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

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

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        Ort::AllocatedStringPtr input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
        input_name_ = input_name_alloc.get();

        Ort::AllocatedStringPtr output_name_alloc = session_->GetOutputNameAllocated(0, allocator_);
        output_name_ = output_name_alloc.get();

        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();

        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor_info.GetShape();

        std::string mode_str = use_coreml ? "CoreML ANE" : "ARM Neon CPU SIMD";
        std::string msg = "Initialized ONNX detector (" + mode_str + " @ " + std::to_string(target_res) + "x" + std::to_string(target_res) + "): " + model_path;
        Logger::getInstance().info("YoloDetector", msg);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().critical("YoloDetector", std::string("Failed to load model file: ") + e.what());
        return false;
    }
}

void YoloDetector::preprocess(const cv::Mat& frame, std::vector<float>& input_tensor_values, float& rx, float& ry, float& pad_w, float& pad_h) {
    int orig_w = frame.cols;
    int orig_h = frame.rows;

    float r = std::min(static_cast<float>(input_width_) / orig_w, static_cast<float>(input_height_) / orig_h);
    int new_w = static_cast<int>(std::round(orig_w * r));
    int new_h = static_cast<int>(std::round(orig_h * r));

    pad_w = static_cast<float>(input_width_ - new_w) / 2.0f;
    pad_h = static_cast<float>(input_height_ - new_h) / 2.0f;

    rx = r;
    ry = r;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(static_cast<int>(pad_w), static_cast<int>(pad_h), new_w, new_h)));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false, CV_32F);

    input_tensor_values.assign((float*)blob.data, (float*)blob.data + blob.total());
}

std::vector<DetectionBox> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<DetectionBox> detections;
    if (!session_ || frame.empty()) return detections;

    std::vector<float> input_tensor_values;
    float rx, ry, pad_w, pad_h;
    preprocess(frame, input_tensor_values, rx, ry, pad_w, pad_h);

    std::vector<int64_t> input_dims = {1, 3, input_height_, input_width_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size()
    );

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );
    } catch (const std::exception& e) {
        Logger::getInstance().error("YoloDetector", std::string("ONNX Runtime execution error: ") + e.what());
        return detections;
    }

    float* float_data = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();

    int dimensions = 84;
    int num_anchors = 8400;

    bool transposed = false;
    if (shape.size() == 3) {
        if (shape[1] == 84) {
            dimensions = 84;
            num_anchors = shape[2];
            transposed = false;
        } else if (shape[2] == 84) {
            dimensions = 84;
            num_anchors = shape[1];
            transposed = true;
        }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float effective_conf = std::max(0.35f, conf_threshold_);

    for (int i = 0; i < num_anchors; ++i) {
        float cx, cy, w, h;
        float max_score = 0.0f;
        int max_class_id = -1;

        if (!transposed) {
            cx = float_data[0 * num_anchors + i];
            cy = float_data[1 * num_anchors + i];
            w  = float_data[2 * num_anchors + i];
            h  = float_data[3 * num_anchors + i];

            for (int c = 4; c < dimensions; ++c) {
                float score = float_data[c * num_anchors + i];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c - 4;
                }
            }
        } else {
            int stride = i * dimensions;
            cx = float_data[stride + 0];
            cy = float_data[stride + 1];
            w  = float_data[stride + 2];
            h  = float_data[stride + 3];

            for (int c = 4; c < dimensions; ++c) {
                float score = float_data[stride + c];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c - 4;
                }
            }
        }

        // STRICT FILTERING: ACCEPT ONLY CLASS 0 (person) AND CLASS 2 (car)
        if ((max_class_id == 0 || max_class_id == 2) && max_score >= effective_conf) {
            float x1 = (cx - w / 2.0f - pad_w) / rx;
            float y1 = (cy - h / 2.0f - pad_h) / ry;
            float bw = w / rx;
            float bh = h / ry;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(frame.cols - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(frame.rows - 1)));
            bw = std::max(1.0f, std::min(bw, static_cast<float>(frame.cols - x1)));
            bh = std::max(1.0f, std::min(bh, static_cast<float>(frame.rows - y1)));

            boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(bw), static_cast<int>(bh)));
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }

    float effective_nms = std::min(0.35f, nms_threshold_);
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, effective_conf, effective_nms, nms_indices);

    std::vector<cv::Rect> final_boxes;
    std::vector<float> final_confs;
    std::vector<int> final_classes;

    for (int idx : nms_indices) {
        cv::Rect cur_box = boxes[idx];
        float cur_conf = confidences[idx];
        bool is_duplicate = false;

        for (size_t k = 0; k < final_boxes.size(); ++k) {
            cv::Rect inter = cur_box & final_boxes[k];
            float inter_area = inter.area();
            float min_area = std::min(cur_box.area(), final_boxes[k].area());

            if (min_area > 0 && (inter_area / min_area) > 0.50f) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            final_boxes.push_back(cur_box);
            final_confs.push_back(cur_conf);
            final_classes.push_back(class_ids[idx]);
        }
    }

    for (size_t i = 0; i < final_boxes.size(); ++i) {
        DetectionBox det;
        det.box = cv::Rect2f(final_boxes[i]);
        det.confidence = final_confs[i];
        det.class_id = final_classes[i];
        det.label = (det.class_id == 0) ? "person" : (det.class_id == 2 ? "car" : "object");
        detections.push_back(det);
    }

    return detections;
}

std::vector<DetectionBox> YoloDetector::detect_batch_simd(const std::vector<cv::Mat>& frames) {
    std::vector<DetectionBox> all_detections;
    if (frames.empty()) return all_detections;

    for (const auto& frame : frames) {
        std::vector<DetectionBox> dets = detect(frame);
        all_detections.insert(all_detections.end(), dets.begin(), dets.end());
    }

    return all_detections;
}
