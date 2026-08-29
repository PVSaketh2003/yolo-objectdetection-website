#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "httplib.h"
#include "yolo_detector.hpp"
#include "object_tracker.hpp"
#include "logger.hpp"
#include "tiling.hpp"
#include "session_manager.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

std::vector<std::unique_ptr<YoloDetector>> g_detector_pool;
std::atomic<bool> g_running(true);

std::string get_session_id_from_req(const httplib::Request& req) {
    if (req.has_header("X-Session-ID")) {
        return req.get_header_value("X-Session-ID");
    }
    if (req.has_param("sid")) {
        return req.get_param_value("sid");
    }
    return "sess_default";
}

void process_session_frame(std::shared_ptr<SessionState> session) {
    std::string target_source, target_type;
    float cur_conf, cur_nms;
    int cur_selected_id;
    bool cur_playing;

    {
        std::lock_guard<std::mutex> lock(session->mtx);
        target_source = session->video_source;
        target_type = session->source_type;
        cur_conf = session->conf_threshold;
        cur_nms = session->nms_threshold;
        cur_selected_id = session->selected_track_id;
        cur_playing = session->is_playing;
    }

    if (!cur_playing) return;

    // Open video source if changed
    if (target_source != session->current_loaded_source) {
        if (session->cap.isOpened()) session->cap.release();

        if (target_type == "webcam") {
            int cam_idx = 0;
            try { cam_idx = std::stoi(target_source); } catch (...) { cam_idx = 0; }
            session->cap.open(cam_idx);
        } else if (target_type == "rtsp" || target_source.rfind("rtsp://", 0) == 0 || target_source.rfind("http://", 0) == 0 || target_source.rfind("https://", 0) == 0) {
            session->cap.open(target_source, cv::CAP_FFMPEG);
        } else {
            std::vector<std::string> candidates = {
                target_source,
                "/Users/pvsairamsaketh/Documents/objectProject/" + target_source,
                "/Users/pvsairamsaketh/Documents/objectProject/test/15690486_1920_1080_25fps.mp4",
                "test/15690486_1920_1080_25fps.mp4"
            };
            for (const auto& path : candidates) {
                if (session->cap.open(path)) break;
            }
        }
        session->current_loaded_source = target_source;
        session->tracker.reset();
        session->frame_counter = 0;
        session->cached_detections.clear();
    }

    cv::Mat frame;
    if (!session->cap.isOpened() || !session->cap.read(frame) || frame.empty()) {
        if (target_type != "webcam" && target_type != "rtsp" && session->cap.isOpened()) {
            session->cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            session->tracker.reset();
            session->frame_counter = 0;
            session->cached_detections.clear();
        }
        return;
    }

    session->frame_counter++;

    g_detector_pool[0]->set_conf_threshold(cur_conf);
    g_detector_pool[0]->set_nms_threshold(cur_nms);

    std::vector<DetectionBox> detections;
    float inf_time = session->inference_ms;

    int cadence = 2;
    if (session->frame_counter % cadence == 1 || session->cached_detections.empty()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        detections = g_detector_pool[0]->detect(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        inf_time = std::chrono::duration<float, std::milli>(t1 - t0).count();
        session->cached_detections = detections;
    } else {
        detections = session->cached_detections;
    }

    // BYTETRACK 8-STATE KALMAN MOTION FILTER UPDATE
    std::vector<TrackedObject> tracks = session->tracker.update(detections);

    float frame_delay_ms = (inf_time / cadence) + 5.0f;
    float current_fps = (frame_delay_ms > 0.0f) ? (1000.0f / frame_delay_ms) : 30.0f;
    if (current_fps > 60.0f) current_fps = 60.0f;

    cv::Mat display_frame = frame.clone();
    cv::Mat crop_mat;
    bool found_selected = false;

    for (const auto& track : tracks) {
        bool is_selected = (track.track_id == cur_selected_id);
        cv::Scalar box_color = is_selected ? cv::Scalar(255, 0, 220) : (track.label == "car" ? cv::Scalar(255, 180, 0) : cv::Scalar(0, 220, 255));
        int thickness = is_selected ? 4 : 2;

        cv::Point p1(static_cast<int>(track.box.x), static_cast<int>(track.box.y));
        cv::Point p2(static_cast<int>(track.box.x + track.box.width), static_cast<int>(track.box.y + track.box.height));
        cv::rectangle(display_frame, p1, p2, box_color, thickness);

        std::string label_str = "#" + std::to_string(track.track_id) + " " + track.label + " " +
                                std::to_string(static_cast<int>(track.confidence * 100)) + "%";

        int base_line = 0;
        cv::Size text_size = cv::getTextSize(label_str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
        cv::Rect label_rect(p1.x, std::max(0, p1.y - text_size.height - 8), text_size.width + 10, text_size.height + 8);

        cv::rectangle(display_frame, label_rect, box_color, -1);
        cv::putText(display_frame, label_str, cv::Point(p1.x + 5, label_rect.y + text_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        for (size_t k = 1; k < track.trajectory.size(); ++k) {
            cv::line(display_frame, track.trajectory[k - 1], track.trajectory[k], box_color, 1, cv::LINE_AA);
        }

        if (is_selected) {
            found_selected = true;
            int pad_x = static_cast<int>(track.box.width * 0.15f);
            int pad_y = static_cast<int>(track.box.height * 0.15f);

            int cx1 = std::max(0, static_cast<int>(track.box.x) - pad_x);
            int cy1 = std::max(0, static_cast<int>(track.box.y) - pad_y);
            int cx2 = std::min(frame.cols, static_cast<int>(track.box.x + track.box.width) + pad_x);
            int cy2 = std::min(frame.rows, static_cast<int>(track.box.y + track.box.height) + pad_y);

            if (cx2 > cx1 && cy2 > cy1) {
                crop_mat = frame(cv::Rect(cx1, cy1, cx2 - cx1, cy2 - cy1)).clone();
                cv::rectangle(crop_mat, cv::Point(0, 0), cv::Point(crop_mat.cols - 1, crop_mat.rows - 1),
                              cv::Scalar(255, 0, 220), 3);
            }
        }
    }

    // DRAW PRESENTABLE FPS & LATENCY MS HUD BADGE
    {
        std::ostringstream hud_stream;
        hud_stream << std::fixed << std::setprecision(2)
                   << "FPS: " << current_fps << " | LATENCY: " << inf_time << " ms";
        std::string hud_text = hud_stream.str();

        int base = 0;
        cv::Size sz = cv::getTextSize(hud_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
        int margin = 15;
        int bx = display_frame.cols - sz.width - margin - 20;
        int by = margin + 10;

        cv::Rect bg_r(bx, by, sz.width + 20, sz.height + 14);
        cv::Mat roi = display_frame(bg_r);
        cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(10, 15, 25));
        double alpha = 0.75;
        cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);

        cv::rectangle(display_frame, bg_r, cv::Scalar(0, 240, 255), 1, cv::LINE_AA);
        cv::putText(display_frame, hud_text, cv::Point(bx + 10, by + sz.height + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 75};
    std::vector<uchar> frame_jpeg, crop_jpeg;

    cv::Mat send_mat = display_frame;
    if (display_frame.cols > 960) {
        double scale = 960.0 / display_frame.cols;
        cv::resize(display_frame, send_mat, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    cv::imencode(".jpg", send_mat, frame_jpeg, params);

    if (found_selected && !crop_mat.empty()) {
        cv::Mat send_crop = crop_mat;
        if (crop_mat.cols > 480) {
            double scale = 480.0 / crop_mat.cols;
            cv::resize(crop_mat, send_crop, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        cv::imencode(".jpg", send_crop, crop_jpeg, params);
    }

    {
        std::lock_guard<std::mutex> lock(session->mtx);
        session->current_frame = display_frame;
        session->current_frame_jpeg = std::move(frame_jpeg);
        session->current_crop_jpeg = std::move(crop_jpeg);
        session->current_tracks = std::move(tracks);
        session->fps = current_fps;
        session->inference_ms = inf_time;
    }
}

void process_loop() {
    Logger::getInstance().info("Engine", "Multi-tenant session processing loop started");

    while (g_running) {
        auto sessions = SessionManager::getInstance().get_all_sessions();
        if (sessions.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        for (auto& session : sessions) {
            if (session) {
                process_session_frame(session);
            }
        }

        SessionManager::getInstance().purge_expired_sessions(30);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void setup_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, X-Session-ID");
}

int main() {
    Logger::getInstance().init("logs/app.log");
    Logger::getInstance().info("Main", "YOLO26 C++ Multi-Tenant Engine starting up...");

    std::vector<std::string> candidate_paths = {
        "backend/models/yolo26s.onnx",
        "models/yolo26s.onnx",
        "../models/yolo26s.onnx",
        "../../backend/models/yolo26s.onnx",
        "yolo26s.onnx"
    };

    std::string model_path = "";
    for (const auto& path : candidate_paths) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            model_path = path;
            break;
        }
    }

    if (model_path.empty()) {
        Logger::getInstance().critical("Main", "Could not locate yolo26s.onnx model file!");
        return 1;
    }

    auto det0 = std::make_unique<YoloDetector>();
    if (!det0->init(model_path, 0.45f, 0.35f, true, 416)) {
        Logger::getInstance().critical("Main", "Failed to initialize primary CoreML detector instance");
        return 1;
    }
    g_detector_pool.push_back(std::move(det0));

    try {
        fs::create_directories("uploads");
        fs::create_directories("uploads/chunks");
    } catch (...) {}

    std::thread proc_thread(process_loop);

    httplib::Server svr;
    svr.set_payload_max_length(1024 * 1024 * 1024);

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        setup_cors(res);
        res.status = 200;
    });

    svr.Post("/api/upload_chunk", [](const httplib::Request& req, httplib::Response& res) {
        setup_cors(res);
        try {
            std::string sid = get_session_id_from_req(req);
            auto session = SessionManager::getInstance().get_session(sid);

            if (!req.form.has_file("chunk")) {
                res.status = 400;
                res.set_content(json({{"error", "Missing chunk data"}}).dump(), "application/json");
                return;
            }

            std::string upload_id = req.form.get_field("upload_id");
            std::string filename = req.form.get_field("filename");
            int chunk_index = std::stoi(req.form.get_field("chunk_index"));
            int total_chunks = std::stoi(req.form.get_field("total_chunks"));

            if (upload_id.empty()) upload_id = "up_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            if (filename.empty()) filename = "video.mp4";

            for (char& c : filename) {
                if (!isalnum(c) && c != '.' && c != '_' && c != '-') c = '_';
            }

            fs::create_directories("uploads/" + sid + "/chunks");
            const auto chunk_data = req.form.get_file("chunk");
            std::string chunk_file_path = "uploads/" + sid + "/chunks/" + upload_id + "_part_" + std::to_string(chunk_index);

            std::ofstream ofs(chunk_file_path, std::ios::binary);
            if (!ofs.is_open()) {
                res.status = 500;
                res.set_content(json({{"error", "Failed to write chunk"}}).dump(), "application/json");
                return;
            }

            ofs.write(chunk_data.content.data(), chunk_data.content.size());
            ofs.close();

            bool all_ready = true;
            for (int i = 0; i < total_chunks; ++i) {
                std::string p = "uploads/" + sid + "/chunks/" + upload_id + "_part_" + std::to_string(i);
                if (!fs::exists(p)) {
                    all_ready = false;
                    break;
                }
            }

            if (all_ready) {
                std::string final_path = "uploads/" + sid + "/sec_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "_" + filename;
                std::ofstream final_ofs(final_path, std::ios::binary);

                for (int i = 0; i < total_chunks; ++i) {
                    std::string p = "uploads/" + sid + "/chunks/" + upload_id + "_part_" + std::to_string(i);
                    std::ifstream ifs(p, std::ios::binary);
                    final_ofs << ifs.rdbuf();
                    ifs.close();
                    fs::remove(p);
                }

                final_ofs.close();
                fs::permissions(final_path, fs::perms::owner_all, fs::perm_options::replace);

                Logger::getInstance().info("UploadChunkAPI", "Private session video upload completed -> " + final_path);

                {
                    std::lock_guard<std::mutex> lock(session->mtx);
                    session->video_source = final_path;
                    session->source_type = "file";
                    session->selected_track_id = -1;
                }
                session->tracker.reset();

                res.set_content(json({
                    {"status", "complete"},
                    {"video_source", final_path}
                }).dump(), "application/json");
            } else {
                res.set_content(json({
                    {"status", "chunk_received"},
                    {"chunk_index", chunk_index}
                }).dump(), "application/json");
            }

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Get("/api/status", [](const httplib::Request& req, httplib::Response& res) {
        setup_cors(res);
        std::string sid = get_session_id_from_req(req);
        auto session = SessionManager::getInstance().get_session(sid);

        json j;
        {
            std::lock_guard<std::mutex> lock(session->mtx);
            j["fps"] = std::round(session->fps * 100.0f) / 100.0f;
            j["inference_ms"] = std::round(session->inference_ms * 100.0f) / 100.0f;
            j["selected_track_id"] = session->selected_track_id;
            j["source_type"] = session->source_type;
            j["video_source"] = session->video_source;
            j["conf_threshold"] = std::round(session->conf_threshold * 100.0f) / 100.0f;
            j["nms_threshold"] = std::round(session->nms_threshold * 100.0f) / 100.0f;
            j["tiling_mode"] = session->tiling_mode;
            j["is_playing"] = session->is_playing;

            json tracks_arr = json::array();
            for (const auto& trk : session->current_tracks) {
                json t_obj;
                t_obj["track_id"] = trk.track_id;
                t_obj["label"] = trk.label;
                t_obj["confidence"] = std::round(trk.confidence * 100.0f) / 100.0f;
                t_obj["class_id"] = trk.class_id;
                t_obj["box"] = {
                    {"x", std::round(trk.box.x)},
                    {"y", std::round(trk.box.y)},
                    {"w", std::round(trk.box.width)},
                    {"h", std::round(trk.box.height)}
                };
                t_obj["velocity"] = {
                    {"dx", std::round(trk.velocity.x * 100.0f) / 100.0f},
                    {"dy", std::round(trk.velocity.y * 100.0f) / 100.0f}
                };
                t_obj["age"] = trk.age;
                tracks_arr.push_back(t_obj);
            }
            j["tracks"] = tracks_arr;
        }

        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/api/select_track", [](const httplib::Request& req, httplib::Response& res) {
        setup_cors(res);
        try {
            std::string sid = get_session_id_from_req(req);
            auto session = SessionManager::getInstance().get_session(sid);

            auto body = json::parse(req.body);
            int track_id = body.value("track_id", -1);
            {
                std::lock_guard<std::mutex> lock(session->mtx);
                session->selected_track_id = track_id;
                session->current_crop_jpeg.clear();
            }
            res.set_content(json({{"status", "ok"}, {"selected_track_id", track_id}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/api/settings", [](const httplib::Request& req, httplib::Response& res) {
        setup_cors(res);
        try {
            std::string sid = get_session_id_from_req(req);
            auto session = SessionManager::getInstance().get_session(sid);

            auto body = json::parse(req.body);
            {
                std::lock_guard<std::mutex> lock(session->mtx);
                if (body.contains("conf_threshold")) session->conf_threshold = body["conf_threshold"];
                if (body.contains("nms_threshold")) session->nms_threshold = body["nms_threshold"];
                if (body.contains("tiling_mode")) session->tiling_mode = body["tiling_mode"];
                if (body.contains("is_playing")) session->is_playing = body["is_playing"];
            }
            res.set_content(json({{"status", "ok"}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/api/source", [](const httplib::Request& req, httplib::Response& res) {
        setup_cors(res);
        try {
            std::string sid = get_session_id_from_req(req);
            auto session = SessionManager::getInstance().get_session(sid);

            auto body = json::parse(req.body);
            std::string src = body.value("video_source", "test/15690486_1920_1080_25fps.mp4");
            std::string type = body.value("source_type", "test_video");
            {
                std::lock_guard<std::mutex> lock(session->mtx);
                session->video_source = src;
                session->source_type = type;
                session->selected_track_id = -1;
            }
            session->tracker.reset();
            res.set_content(json({{"status", "ok"}, {"video_source", src}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Get("/api/stream", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
        res.set_header("X-Accel-Buffering", "no");
        std::string sid = get_session_id_from_req(req);
        auto session = SessionManager::getInstance().get_session(sid);

        res.set_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [session](size_t offset, httplib::DataSink& sink) {
                if (!sink.is_writable()) return false;

                std::vector<uchar> buf;
                {
                    std::lock_guard<std::mutex> lock(session->mtx);
                    buf = session->current_frame_jpeg;
                }

                if (buf.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    return true;
                }

                std::ostringstream header;
                header << "--frame\r\n"
                       << "Content-Type: image/jpeg\r\n"
                       << "Content-Length: " << buf.size() << "\r\n\r\n";

                std::string header_str = header.str();
                if (!sink.write(header_str.data(), header_str.size())) return false;
                if (!sink.write(reinterpret_cast<const char*>(buf.data()), buf.size())) return false;
                if (!sink.write("\r\n", 2)) return false;

                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                return true;
            }
        );
    });

    svr.Get("/api/crop", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
        res.set_header("X-Accel-Buffering", "no");
        std::string sid = get_session_id_from_req(req);
        auto session = SessionManager::getInstance().get_session(sid);

        res.set_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [session](size_t offset, httplib::DataSink& sink) {
                if (!sink.is_writable()) return false;

                std::vector<uchar> buf;
                {
                    std::lock_guard<std::mutex> lock(session->mtx);
                    buf = session->current_crop_jpeg;
                }

                if (buf.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    return true;
                }

                std::ostringstream header;
                header << "--frame\r\n"
                       << "Content-Type: image/jpeg\r\n"
                       << "Content-Length: " << buf.size() << "\r\n\r\n";

                std::string header_str = header.str();
                if (!sink.write(header_str.data(), header_str.size())) return false;
                if (!sink.write(reinterpret_cast<const char*>(buf.data()), buf.size())) return false;
                if (!sink.write("\r\n", 2)) return false;

                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                return true;
            }
        );
    });

    int port = 8080;
    Logger::getInstance().info("Main", "Server listening on http://0.0.0.0:" + std::to_string(port));

    while (g_running && !svr.listen("0.0.0.0", port)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    g_running = false;
    if (proc_thread.joinable()) proc_thread.join();
    return 0;
}
