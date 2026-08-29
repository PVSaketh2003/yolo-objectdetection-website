#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <opencv2/opencv.hpp>

/**
 * ByteTrack 8-State Kalman Filter for Multi-Object Tracking
 * State: [x, y, a, h, vx, vy, va, vh]^T
 *  (x, y): center position of bounding box
 *  a: aspect ratio (w / h)
 *  h: height of bounding box
 *  (vx, vy, va, vh): respective velocities
 */
class ByteKalmanFilter {
public:
    ByteKalmanFilter() {
        kf_ = cv::KalmanFilter(8, 4, 0, CV_32F);

        // Transition Matrix F (dt = 1.0)
        kf_.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
        for (int i = 0; i < 4; ++i) {
            kf_.transitionMatrix.at<float>(i, i + 4) = 1.0f;
        }

        // Measurement Matrix H
        kf_.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
        for (int i = 0; i < 4; ++i) {
            kf_.measurementMatrix.at<float>(i, i) = 1.0f;
        }

        // Process Noise Covariance Q
        cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
        for (int i = 4; i < 8; ++i) {
            kf_.processNoiseCov.at<float>(i, i) = 1e-1;
        }

        // Measurement Noise Covariance R
        cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));

        // Posteriori Error Covariance P
        cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(10.0));
    }

    void init(const cv::Rect2f& box) {
        float cx = box.x + box.width / 2.0f;
        float cy = box.y + box.height / 2.0f;
        float aspect = (box.height > 0.0f) ? (box.width / box.height) : 1.0f;
        float height = box.height;

        kf_.statePost.at<float>(0) = cx;
        kf_.statePost.at<float>(1) = cy;
        kf_.statePost.at<float>(2) = aspect;
        kf_.statePost.at<float>(3) = height;
        kf_.statePost.at<float>(4) = 0.0f;
        kf_.statePost.at<float>(5) = 0.0f;
        kf_.statePost.at<float>(6) = 0.0f;
        kf_.statePost.at<float>(7) = 0.0f;
    }

    cv::Rect2f predict() {
        cv::Mat prediction = kf_.predict();
        float cx = prediction.at<float>(0);
        float cy = prediction.at<float>(1);
        float aspect = std::max(0.1f, prediction.at<float>(2));
        float height = std::max(1.0f, prediction.at<float>(3));
        float width = aspect * height;

        return cv::Rect2f(cx - width / 2.0f, cy - height / 2.0f, width, height);
    }

    cv::Rect2f update(const cv::Rect2f& box) {
        float cx = box.x + box.width / 2.0f;
        float cy = box.y + box.height / 2.0f;
        float aspect = (box.height > 0.0f) ? (box.width / box.height) : 1.0f;
        float height = box.height;

        cv::Mat measurement = (cv::Mat_<float>(4, 1) << cx, cy, aspect, height);
        cv::Mat estimated = kf_.correct(measurement);

        float est_cx = estimated.at<float>(0);
        float est_cy = estimated.at<float>(1);
        float est_aspect = std::max(0.1f, estimated.at<float>(2));
        float est_height = std::max(1.0f, estimated.at<float>(3));
        float est_width = est_aspect * est_height;

        return cv::Rect2f(est_cx - est_width / 2.0f, est_cy - est_height / 2.0f, est_width, est_height);
    }

    cv::Point2f get_velocity() const {
        return cv::Point2f(kf_.statePost.at<float>(4), kf_.statePost.at<float>(5));
    }

private:
    cv::KalmanFilter kf_;
};

#endif // KALMAN_FILTER_HPP
