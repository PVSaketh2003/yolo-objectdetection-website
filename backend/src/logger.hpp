#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <chrono>

enum class LogLevel {
    INFO,
    WARNING,
    LOG_ERROR,
    CRITICAL
};

struct LogEntry {
    std::string timestamp;
    LogLevel level;
    std::string message;
    std::string module;
};

class Logger {
public:
    static Logger& getInstance();

    void init(const std::string& log_file_path = "logs/app.log");
    void log(LogLevel level, const std::string& module, const std::string& message);
    
    void info(const std::string& module, const std::string& message);
    void warn(const std::string& module, const std::string& message);
    void error(const std::string& module, const std::string& message);
    void critical(const std::string& module, const std::string& message);

    std::vector<LogEntry> get_recent_logs(size_t max_count = 50);

private:
    Logger() = default;
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::mutex mutex_;
    std::ofstream log_file_;
    std::vector<LogEntry> ring_buffer_;
    static const size_t MAX_RING_BUFFER_SIZE = 200;

    std::string get_current_timestamp();
    std::string level_to_string(LogLevel level);
};

#endif // LOGGER_HPP
