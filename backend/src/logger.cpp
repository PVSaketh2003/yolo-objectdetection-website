#include "logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::~Logger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::init(const std::string& log_file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        fs::path p(log_file_path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        log_file_.open(log_file_path, std::ios::out | std::ios::app);
        if (log_file_.is_open()) {
            log_file_ << "\n--- Logging Session Started [" << get_current_timestamp() << "] ---\n";
            log_file_.flush();
        }
    } catch (const std::exception& e) {
        std::cerr << "[Logger Init Exception] " << e.what() << std::endl;
    }
}

std::string Logger::get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::LOG_ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& module, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string ts = get_current_timestamp();
    std::string lvl_str = level_to_string(level);

    LogEntry entry{ts, level, message, module};

    // Keep ring buffer bounded
    if (ring_buffer_.size() >= MAX_RING_BUFFER_SIZE) {
        ring_buffer_.erase(ring_buffer_.begin());
    }
    ring_buffer_.push_back(entry);

    // Colorized console output
    std::string color_code = "\033[0m"; // Reset
    if (level == LogLevel::INFO) color_code = "\033[36m";       // Cyan
    else if (level == LogLevel::WARNING) color_code = "\033[33m"; // Yellow
    else if (level == LogLevel::LOG_ERROR) color_code = "\033[31m";  // Red
    else if (level == LogLevel::CRITICAL) color_code = "\033[35m"; // Magenta

    std::cout << color_code << "[" << ts << "] [" << lvl_str << "] [" << module << "] " << message << "\033[0m\n";

    // Write to log file if open
    if (log_file_.is_open()) {
        log_file_ << "[" << ts << "] [" << lvl_str << "] [" << module << "] " << message << "\n";
        log_file_.flush();
    }
}

void Logger::info(const std::string& module, const std::string& message) {
    log(LogLevel::INFO, module, message);
}

void Logger::warn(const std::string& module, const std::string& message) {
    log(LogLevel::WARNING, module, message);
}

void Logger::error(const std::string& module, const std::string& message) {
    log(LogLevel::LOG_ERROR, module, message);
}

void Logger::critical(const std::string& module, const std::string& message) {
    log(LogLevel::CRITICAL, module, message);
}

std::vector<LogEntry> Logger::get_recent_logs(size_t max_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ring_buffer_.size() <= max_count) {
        return ring_buffer_;
    }
    return std::vector<LogEntry>(ring_buffer_.end() - max_count, ring_buffer_.end());
}
