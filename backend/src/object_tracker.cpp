#include "object_tracker.hpp"
#include "lapjv.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cmath>

ObjectTracker::ObjectTracker(float iou_threshold, int max_missing)
    : iou_threshold_(iou_threshold), max_missing_(30) {}

float ObjectTracker::compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) const {
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

std::vector<TrackedObject> ObjectTracker::update(const std::vector<DetectionBox>& detections) {
    try {
        int num_tracks = static_cast<int>(active_tracks_.size());

        // STEP 1: KALMAN MOTION PREDICTION (ACCURATE REAL-TIME SMOOTH BOX MOTION)
        for (auto& trk : active_tracks_) {
            cv::Rect2f pred_box = trk.kalman.predict();
            trk.box = pred_box;
        }

        // STEP 2: BYTETRACK PARTITIONING INTO HIGH AND LOW CONFIDENCE DETECTIONS
        std::vector<DetectionBox> high_dets;
        std::vector<DetectionBox> low_dets;

        for (const auto& det : detections) {
            if (det.confidence >= 0.45f) {
                high_dets.push_back(det);
            } else if (det.confidence >= 0.15f) {
                low_dets.push_back(det);
            }
        }

        int num_high = static_cast<int>(high_dets.size());
        std::vector<bool> matched_tracks(num_tracks, false);
        std::vector<bool> matched_high(num_high, false);

        // STAGE 1: MATCH ACTIVE TRACKS WITH HIGH-CONFIDENCE DETECTIONS USING LAPJV
        if (num_tracks > 0 && num_high > 0) {
            std::vector<std::vector<float>> cost_matrix(num_tracks, std::vector<float>(num_high, 1e6f));

            for (int t = 0; t < num_tracks; ++t) {
                for (int d = 0; d < num_high; ++d) {
                    if (active_tracks_[t].class_id == high_dets[d].class_id) {
                        float iou = compute_iou(active_tracks_[t].box, high_dets[d].box);
                        if (iou >= 0.15f) {
                            cost_matrix[t][d] = 1.0f - iou;
                        }
                    }
                }
            }

            float max_cost = 1.0f - iou_threshold_;
            std::vector<int> assignments = LAPJV::solve(cost_matrix, max_cost);

            for (int t = 0; t < num_tracks; ++t) {
                if (t < static_cast<int>(assignments.size())) {
                    int d = assignments[t];
                    if (d >= 0 && d < num_high && !matched_high[d]) {
                        matched_tracks[t] = true;
                        matched_high[d] = true;

                        // Kalman Measurement Update
                        cv::Rect2f est_box = active_tracks_[t].kalman.update(high_dets[d].box);
                        active_tracks_[t].box = est_box;
                        active_tracks_[t].confidence = high_dets[d].confidence;
                        active_tracks_[t].age += 1;
                        active_tracks_[t].missing_frames = 0;
                        active_tracks_[t].velocity = active_tracks_[t].kalman.get_velocity();

                        cv::Point2f center(est_box.x + est_box.width / 2.0f, est_box.y + est_box.height / 2.0f);
                        active_tracks_[t].trajectory.push_back(center);
                        if (active_tracks_[t].trajectory.size() > 30) {
                            active_tracks_[t].trajectory.erase(active_tracks_[t].trajectory.begin());
                        }
                    }
                }
            }
        }

        // STAGE 2: MATCH REMAINING UNASSIGNED TRACKS WITH LOW-CONFIDENCE DETECTIONS
        int num_low = static_cast<int>(low_dets.size());
        std::vector<int> unassigned_tracks;
        for (int t = 0; t < num_tracks; ++t) {
            if (!matched_tracks[t]) unassigned_tracks.push_back(t);
        }

        if (!unassigned_tracks.empty() && num_low > 0) {
            int num_unassigned = static_cast<int>(unassigned_tracks.size());
            std::vector<std::vector<float>> cost_matrix_low(num_unassigned, std::vector<float>(num_low, 1e6f));

            for (int i = 0; i < num_unassigned; ++i) {
                int t = unassigned_tracks[i];
                for (int d = 0; d < num_low; ++d) {
                    if (active_tracks_[t].class_id == low_dets[d].class_id) {
                        float iou = compute_iou(active_tracks_[t].box, low_dets[d].box);
                        if (iou >= 0.20f) {
                            cost_matrix_low[i][d] = 1.0f - iou;
                        }
                    }
                }
            }

            std::vector<int> low_assignments = LAPJV::solve(cost_matrix_low, 0.70f);

            for (int i = 0; i < num_unassigned; ++i) {
                if (i < static_cast<int>(low_assignments.size())) {
                    int d = low_assignments[i];
                    if (d >= 0 && d < num_low) {
                        int t = unassigned_tracks[i];
                        matched_tracks[t] = true;

                        cv::Rect2f est_box = active_tracks_[t].kalman.update(low_dets[d].box);
                        active_tracks_[t].box = est_box;
                        active_tracks_[t].confidence = low_dets[d].confidence;
                        active_tracks_[t].age += 1;
                        active_tracks_[t].missing_frames = 0;
                        active_tracks_[t].velocity = active_tracks_[t].kalman.get_velocity();
                    }
                }
            }
        }

        // STAGE 3: TRACK BUFFER PERSISTENCE (30 FRAMES RE-ID PERSISTENCE)
        for (int t = 0; t < num_tracks; ++t) {
            if (!matched_tracks[t]) {
                active_tracks_[t].missing_frames += 1;
                active_tracks_[t].velocity = active_tracks_[t].kalman.get_velocity();
            }
        }

        // REMOVE EXPIRED TRACKS AFTER 30 MISSING FRAMES
        active_tracks_.erase(
            std::remove_if(active_tracks_.begin(), active_tracks_.end(), [](const TrackedObject& obj) {
                return obj.missing_frames > 30; // BYTETRACK 30-FRAME TRACK BUFFER
            }),
            active_tracks_.end()
        );

        // CREATE NEW TRACKS FOR UNMATCHED HIGH-CONFIDENCE DETECTIONS
        for (int d = 0; d < num_high; ++d) {
            if (!matched_high[d]) {
                bool exists_duplicate = false;
                for (const auto& active_trk : active_tracks_) {
                    float iou = compute_iou(active_trk.box, high_dets[d].box);

                    cv::Point2f c1(active_trk.box.x + active_trk.box.width / 2.0f, active_trk.box.y + active_trk.box.height / 2.0f);
                    cv::Point2f c2(high_dets[d].box.x + high_dets[d].box.width / 2.0f, high_dets[d].box.y + high_dets[d].box.height / 2.0f);
                    float dist = std::hypot(c1.x - c2.x, c1.y - c2.y);

                    if (iou > 0.25f || dist < 30.0f) {
                        exists_duplicate = true;
                        break;
                    }
                }

                if (!exists_duplicate) {
                    TrackedObject new_obj;
                    new_obj.track_id = next_track_id_++;
                    new_obj.box = high_dets[d].box;
                    new_obj.label = high_dets[d].label;
                    new_obj.confidence = high_dets[d].confidence;
                    new_obj.class_id = high_dets[d].class_id;
                    new_obj.age = 1;
                    new_obj.missing_frames = 0;
                    new_obj.velocity = cv::Point2f(0, 0);

                    // Initialize ByteTrack 8-State Kalman Filter
                    new_obj.kalman.init(new_obj.box);

                    cv::Point2f center(
                        new_obj.box.x + new_obj.box.width / 2.0f,
                        new_obj.box.y + new_obj.box.height / 2.0f
                    );
                    new_obj.trajectory.push_back(center);

                    active_tracks_.push_back(new_obj);
                }
            }
        }

        // STRICT TRACK-LEVEL PAIRWISE DEDUPLICATION PASS
        std::sort(active_tracks_.begin(), active_tracks_.end(), [](const TrackedObject& a, const TrackedObject& b) {
            if (a.confidence != b.confidence) return a.confidence > b.confidence;
            return a.age > b.age;
        });

        std::vector<TrackedObject> unique_tracks;
        for (const auto& trk : active_tracks_) {
            bool duplicate = false;
            for (const auto& u_trk : unique_tracks) {
                float iou = compute_iou(trk.box, u_trk.box);
                cv::Point2f c1(trk.box.x + trk.box.width / 2.0f, trk.box.y + trk.box.height / 2.0f);
                cv::Point2f c2(u_trk.box.x + u_trk.box.width / 2.0f, u_trk.box.y + u_trk.box.height / 2.0f);
                float dist = std::hypot(c1.x - c2.x, c1.y - c2.y);

                if (iou > 0.25f || dist < 30.0f) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                unique_tracks.push_back(trk);
            }
        }

        active_tracks_ = std::move(unique_tracks);

    } catch (const std::exception& e) {
        Logger::getInstance().error("ObjectTracker", std::string("ByteTrack update exception caught: ") + e.what());
    }

    return active_tracks_;
}
