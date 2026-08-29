#ifndef TILING_HPP
#define TILING_HPP

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "yolo_detector.hpp"

struct TileRegion {
    cv::Rect rect;
    int tile_id;
};

class TilingEngine {
public:
    TilingEngine() = default;

    static std::vector<TileRegion> generate_tiles(int frame_w, int frame_h, int num_tiles_x = 2, int num_tiles_y = 2, float overlap_pct = 0.20f);

    static std::vector<DetectionBox> detect_with_tiling_pool(
        std::vector<std::unique_ptr<YoloDetector>>& detector_pool,
        const cv::Mat& full_frame,
        int tile_grid_mode = 4,
        float iou_threshold = 0.35f
    );

    // SCALE-GATED SAHI TILING & CROSS-SCALE NMS FUSION
    static std::vector<DetectionBox> detect_with_sahi_tiling(
        std::vector<std::unique_ptr<YoloDetector>>& detector_pool,
        const cv::Mat& full_frame,
        int tile_grid_mode = 4,
        float iou_threshold = 0.35f
    );

private:
    static float calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2);
};

#endif // TILING_HPP
