#include "tiling.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <future>

std::vector<TileRegion> TilingEngine::generate_tiles(int frame_w, int frame_h, int num_tiles_x, int num_tiles_y, float overlap_pct) {
    std::vector<TileRegion> tiles;
    if (frame_w <= 0 || frame_h <= 0) return tiles;

    if (num_tiles_x <= 1 && num_tiles_y <= 1) {
        tiles.push_back({cv::Rect(0, 0, frame_w, frame_h), 0});
        return tiles;
    }

    int tile_w = static_cast<int>(std::ceil(static_cast<float>(frame_w) / num_tiles_x));
    int tile_h = static_cast<int>(std::ceil(static_cast<float>(frame_h) / num_tiles_y));

    int overlap_w = static_cast<int>(tile_w * overlap_pct);
    int overlap_h = static_cast<int>(tile_h * overlap_pct);

    int tile_count = 0;
    for (int y_idx = 0; y_idx < num_tiles_y; ++y_idx) {
        for (int x_idx = 0; x_idx < num_tiles_x; ++x_idx) {
            int x1 = x_idx * tile_w - (x_idx > 0 ? overlap_w : 0);
            int y1 = y_idx * tile_h - (y_idx > 0 ? overlap_h : 0);

            int x2 = (x_idx + 1) * tile_w + ((x_idx < num_tiles_x - 1) ? overlap_w : 0);
            int y2 = (y_idx + 1) * tile_h + ((y_idx < num_tiles_y - 1) ? overlap_h : 0);

            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(frame_w, x2);
            y2 = std::min(frame_h, y2);

            tiles.push_back({cv::Rect(x1, y1, x2 - x1, y2 - y1), tile_count++});
        }
    }

    return tiles;
}

float TilingEngine::calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    cv::Rect2f inter = box1 & box2;
    float inter_area = inter.area();
    if (inter_area <= 0.0f) return 0.0f;

    float union_area = box1.area() + box2.area() - inter_area;
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

std::vector<DetectionBox> TilingEngine::detect_with_tiling_pool(
    std::vector<std::unique_ptr<YoloDetector>>& detector_pool,
    const cv::Mat& full_frame,
    int tile_grid_mode,
    float iou_threshold
) {
    return detect_with_sahi_tiling(detector_pool, full_frame, tile_grid_mode, iou_threshold);
}

// SCALE-GATED SAHI TILING & CROSS-SCALE NMS FUSION
std::vector<DetectionBox> TilingEngine::detect_with_sahi_tiling(
    std::vector<std::unique_ptr<YoloDetector>>& detector_pool,
    const cv::Mat& full_frame,
    int tile_grid_mode,
    float iou_threshold
) {
    std::vector<DetectionBox> all_raw_detections;
    if (full_frame.empty() || detector_pool.empty()) return all_raw_detections;

    int orig_w = full_frame.cols;
    int orig_h = full_frame.rows;

    if (tile_grid_mode <= 1) {
        return detector_pool[0]->detect(full_frame);
    }

    // 1. GENERATE 4 OVERLAPPING SUB-TILES (20% OVERLAP)
    std::vector<TileRegion> tiles = generate_tiles(orig_w, orig_h, 2, 2, 0.20f);

    std::vector<std::future<std::vector<DetectionBox>>> futures;
    futures.reserve(tiles.size());

    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& tile = tiles[i];
        cv::Mat tile_crop = full_frame(tile.rect).clone();
        size_t det_idx = (i + 1) % detector_pool.size();
        if (det_idx == 0 && detector_pool.size() > 1) det_idx = 1;
        YoloDetector* det_ptr = detector_pool[det_idx].get();

        futures.push_back(std::async(std::launch::async, [det_ptr, tile_crop, tile]() {
            std::vector<DetectionBox> tile_dets = det_ptr->detect(tile_crop);
            std::vector<DetectionBox> small_gated_dets;

            for (auto& det : tile_dets) {
                // SCALE GATING RULE: SUB-TILES ONLY DETECT SMALL / DISTANT OBJECTS!
                if (det.class_id == 0) { // Person
                    // Person in sub-tile must be SMALL/DISTANT (height <= 140px in full-frame space)
                    if (det.box.height > 140.0f || det.box.height < 18.0f) continue;
                    float aspect = det.box.height / det.box.width;
                    if (aspect < 1.10f || aspect > 3.8f) continue; // Eliminates crop artifact boxes on car doors!
                } else if (det.class_id == 2) { // Car
                    // Car in sub-tile must be SMALL/DISTANT (width <= 180px in full-frame space)
                    if (det.box.width > 180.0f || det.box.width < 20.0f) continue;
                    float aspect = det.box.width / det.box.height;
                    if (aspect < 1.05f || aspect > 3.8f) continue;
                }

                // Translate local sub-tile coordinates to full-frame space
                det.box.x += tile.rect.x;
                det.box.y += tile.rect.y;

                small_gated_dets.push_back(det);
            }
            return small_gated_dets;
        }));
    }

    for (auto& fut : futures) {
        std::vector<DetectionBox> tile_dets = fut.get();
        all_raw_detections.insert(all_raw_detections.end(), tile_dets.begin(), tile_dets.end());
    }

    // STRICT CROSS-SCALE IOU & IOA FUSION DEDUPLICATION
    std::sort(all_raw_detections.begin(), all_raw_detections.end(), [](const DetectionBox& a, const DetectionBox& b) {
        return a.confidence > b.confidence;
    });

    std::vector<DetectionBox> fused_detections;
    std::vector<bool> suppressed(all_raw_detections.size(), false);

    for (size_t i = 0; i < all_raw_detections.size(); ++i) {
        if (suppressed[i]) continue;

        const auto& best_det = all_raw_detections[i];
        fused_detections.push_back(best_det);

        for (size_t j = i + 1; j < all_raw_detections.size(); ++j) {
            if (suppressed[j]) continue;

            const auto& candidate = all_raw_detections[j];

            float iou = calculate_iou(best_det.box, candidate.box);

            cv::Rect2f inter = best_det.box & candidate.box;
            float inter_area = inter.area();
            float min_area = std::min(best_det.box.area(), candidate.box.area());
            float ioa = (min_area > 0.0f) ? (inter_area / min_area) : 0.0f;

            float cx1 = best_det.box.x + best_det.box.width / 2.0f;
            float cy1 = best_det.box.y + best_det.box.height / 2.0f;
            float cx2 = candidate.box.x + candidate.box.width / 2.0f;
            float cy2 = candidate.box.y + candidate.box.height / 2.0f;
            float center_dist = std::hypot(cx1 - cx2, cy1 - cy2);

            if (iou > 0.25f || ioa > 0.35f || center_dist < 28.0f) {
                suppressed[j] = true;
            }
        }
    }

    return fused_detections;
}
