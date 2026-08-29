#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "yolo_detector.hpp"
#include "kalman_filter.hpp"

struct TrackedObject {
    int track_id;
    cv::Rect2f box;
    std::string label;
    float confidence;
    int class_id;
    int age = 1;
    int missing_frames = 0;
    cv::Point2f velocity{0.0f, 0.0f};
    std::vector<cv::Point2f> trajectory;
    ByteKalmanFilter kalman;
};

class ObjectTracker {
public:
    explicit ObjectTracker(float iou_threshold = 0.30f, int max_missing = 30);
    ~ObjectTracker() = default;

    std::vector<TrackedObject> update(const std::vector<DetectionBox>& detections);

    void reset() {
        active_tracks_.clear();
        next_track_id_ = 1;
    }

    void set_iou_threshold(float val) { iou_threshold_ = val; }
    void set_max_missing(int val) { max_missing_ = val; }

private:
    float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) const;

    float iou_threshold_ = 0.30f;
    int max_missing_ = 30; // BYTETRACK 30-FRAME TRACK BUFFER
    int next_track_id_ = 1;

    std::vector<TrackedObject> active_tracks_;
};

#endif // OBJECT_TRACKER_HPP
