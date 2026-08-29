#ifndef SESSION_MANAGER_HPP
#define SESSION_MANAGER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "object_tracker.hpp"
#include "logger.hpp"

namespace fs = std::filesystem;

struct SessionState {
    std::string session_id;
    std::mutex mtx;

    cv::VideoCapture cap;
    std::string video_source = "test/15690486_1920_1080_25fps.mp4";
    std::string source_type = "test_video";
    std::string current_loaded_source = "";
    bool is_playing = true;

    ObjectTracker tracker{0.30f, 30};

    cv::Mat current_frame;
    std::vector<uchar> current_frame_jpeg;
    std::vector<uchar> current_crop_jpeg;
    std::vector<TrackedObject> current_tracks;

    int selected_track_id = -1;
    float fps = 0.0f;
    float inference_ms = 0.0f;

    float conf_threshold = 0.45f;
    float nms_threshold = 0.35f;
    int tiling_mode = 0;

    std::chrono::system_clock::time_point last_activity = std::chrono::system_clock::now();
};

class SessionManager {
public:
    static SessionManager& getInstance() {
        static SessionManager instance;
        return instance;
    }

    std::shared_ptr<SessionState> get_session(const std::string& raw_sid) {
        std::string sid = sanitize_session_id(raw_sid);

        std::lock_guard<std::mutex> lock(map_mtx_);
        auto it = sessions_.find(sid);
        if (it != sessions_.end()) {
            it->second->last_activity = std::chrono::system_clock::now();
            return it->second;
        }

        auto new_session = std::make_shared<SessionState>();
        new_session->session_id = sid;
        new_session->last_activity = std::chrono::system_clock::now();

        // Create private session uploads directory
        try {
            fs::create_directories("uploads/" + sid);
            fs::permissions("uploads/" + sid, fs::perms::owner_all, fs::perm_options::replace);
        } catch (...) {}

        sessions_[sid] = new_session;
        Logger::getInstance().info("SessionManager", "Created 100% private session state for Session ID: " + sid);
        return new_session;
    }

    void purge_expired_sessions(int max_idle_minutes = 30) {
        std::lock_guard<std::mutex> lock(map_mtx_);
        auto now = std::chrono::system_clock::now();

        for (auto it = sessions_.begin(); it != sessions_.end(); ) {
            auto idle_time = std::chrono::duration_cast<std::chrono::minutes>(now - it->second->last_activity).count();
            if (idle_time >= max_idle_minutes) {
                std::string sid = it->first;
                Logger::getInstance().info("SessionManager", "Purging expired session: " + sid);

                try {
                    fs::remove_all("uploads/" + sid);
                } catch (...) {}

                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<std::shared_ptr<SessionState>> get_all_sessions() {
        std::lock_guard<std::mutex> lock(map_mtx_);
        std::vector<std::shared_ptr<SessionState>> list;
        for (const auto& kv : sessions_) {
            list.push_back(kv.second);
        }
        return list;
    }

private:
    SessionManager() = default;
    ~SessionManager() = default;

    std::string sanitize_session_id(const std::string& sid) const {
        if (sid.empty()) return "sess_default";
        std::string clean = "";
        for (char c : sid) {
            if (std::isalnum(c) || c == '_' || c == '-') clean += c;
        }
        return clean.empty() ? "sess_default" : clean;
    }

    std::mutex map_mtx_;
    std::unordered_map<std::string, std::shared_ptr<SessionState>> sessions_;
};

#endif // SESSION_MANAGER_HPP
