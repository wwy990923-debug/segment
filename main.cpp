#include "yolov8_batch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <filesystem>
namespace fs = std::filesystem;
bool fileExists(const std::string& p) { return fs::exists(p); }
void printVideoInfo(cv::VideoCapture& cap, const std::string& n)
{
    std::cout << n << " - "
        << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
        << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        << ", FPS: " << cap.get(cv::CAP_PROP_FPS)
        << ", Frames: " << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT))
        << std::endl;
}
// Action state to string
std::string actionStateToString(ActionState state) {
    switch (state) {
    case ActionState::IDLE: return "IDLE";
    case ActionState::DETECTED: return "DETECTED";
    case ActionState::OK: return "OK";
    case ActionState::NG: return "NG";
    default: return "UNKNOWN";
    }
}
// Action type to string
std::string actionTypeToString(ActionType type) {
    return (type == ActionType::INSTANT) ? "INSTANT" : "CONTINUOUS";
}
int main()
{
    /* ---------- Model Paths ---------- */
    std::string model1_path = "models/b12.bin"; // View1 - 准备(pad), 吸屏2(pick)
    std::string model2_path = "models/b12.bin"; // View2 - 吸屏1,贴膜,撕膜,按压,棉刷,取支架,按压支架
    std::string model3_path = "models/b12.bin"; // View3 - 放回(zhijia)
    std::string pose_model_path = "models/yolov8n-pose.engine"; // View2 - Pose estimation (NEW)
    /* ---------- Video Paths ---------- */
    std::string video1, video2, video3;
    std::cout << "=== YOLOv8 Triple View Sequential Detection with Pose ===\n";
    std::cout << "View1 video path (准备/吸屏2): ";
    std::getline(std::cin, video1);
    std::cout << "View2 video path (主操作): ";
    std::getline(std::cin, video2);
    std::cout << "View3 video path (放回): ";
    std::getline(std::cin, video3);
    if (video1.empty()) video1 = "D:\\BaiduNetdiskDownload\\video\\test\\Video_20260205013609736_sync.avi";
    if (video2.empty()) video2 = "D:\\BaiduNetdiskDownload\\video\\test\\Video_20260205013612179_sync.avi";
    if (video3.empty()) video3 = "D:\\BaiduNetdiskDownload\\video\\test\\Video_20260205013614014_sync.avi";
    if (!fileExists(video1) || !fileExists(video2) || !fileExists(video3)) {
        std::cerr << "One or more video files missing\n";
        return -1;
    }
    /* ---------- Create Triple View Processor ---------- */
    std::unique_ptr<TripleViewProcessor> processor;
    try {
        // 传入pose模型路径（第4个参数）
        processor = std::make_unique<TripleViewProcessor>(model1_path, model2_path, model3_path, pose_model_path);
        processor->init();
        std::cout << "TripleViewProcessor initialized successfully\n";
        std::cout << "Work Sequence: 准备 -> 吸屏1 -> 贴膜 -> 撕膜 -> 按压 -> 棉刷 -> 取支架 -> 按压支架 -> 放回 -> 吸屏2 -> (Cycle)\n";
        // 显示Pose检测器状态
        if (processor->get_pose_overlay()) {
            std::cout << "[INFO] Pose overlay: ENABLED (pose will be drawn on View2)\n";
        }
        else {
            std::cout << "[INFO] Pose overlay: DISABLED or Pose model not loaded\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return -1;
    }
    /* ---------- Open Videos ---------- */
    cv::VideoCapture cap1(video1), cap2(video2), cap3(video3);
    if (!cap1.isOpened() || !cap2.isOpened() || !cap3.isOpened()) {
        std::cerr << "Cannot open videos\n";
        return -1;
    }
    printVideoInfo(cap1, "View1");
    printVideoInfo(cap2, "View2");
    printVideoInfo(cap3, "View3");
    /* ---------- Create Windows ---------- */
    cv::namedWindow("View1 - 准备/吸屏2", cv::WINDOW_NORMAL);
    cv::namedWindow("View2 - 主操作", cv::WINDOW_NORMAL);
    cv::namedWindow("View3 - 放回", cv::WINDOW_NORMAL);
    cv::namedWindow("Work Sequence Status", cv::WINDOW_NORMAL);
    // 可选：创建独立的Pose显示窗口（如果不使用叠加模式）
    bool show_separate_pose_window = false; // 设为true可单独显示pose窗口
    if (show_separate_pose_window && processor->get_pose_overlay()) {
        cv::namedWindow("View2 - Pose Only", cv::WINDOW_NORMAL);
    }
    cv::resizeWindow("View1 - 准备/吸屏2", 640, 480);
    cv::resizeWindow("View2 - 主操作", 640, 480);
    cv::resizeWindow("View3 - 放回", 640, 480);
    cv::resizeWindow("Work Sequence Status", 700, 600);
    if (show_separate_pose_window && processor->get_pose_overlay()) {
        cv::resizeWindow("View2 - Pose Only", 640, 480);
    }
    cv::moveWindow("View1 - 准备/吸屏2", 50, 50);
    cv::moveWindow("View2 - 主操作", 700, 50);
    cv::moveWindow("View3 - 放回", 1350, 50);
    cv::moveWindow("Work Sequence Status", 50, 600);
    if (show_separate_pose_window && processor->get_pose_overlay()) {
        cv::moveWindow("View2 - Pose Only", 700, 550);
    }
    /* ---------- Main Loop ---------- */
    cv::Mat frame1, frame2, frame3;
    cv::Mat statusImg(600, 700, CV_8UC3, cv::Scalar(30, 30, 30));
    bool paused = false;
    int frameCnt = 0;
    long long totalMs = 0;
    auto action_callback = [](const std::string& action, const std::string& status, const std::string& reason) {
        std::cout << "Action completed: " << action << " - " << status;
        if (!reason.empty()) {
            std::cout << " - Reason: " << reason;
        }
        std::cout << std::endl;
        };
    auto action_manager = processor->get_action_manager();
    action_manager->set_action_callback(action_callback);
    std::cout << "Starting main loop... Press ESC to exit, SPACE to pause, R to reset, P to toggle pose overlay" << std::endl;
    for (;;)
    {
        if (!paused) {
            bool ok1 = cap1.read(frame1), ok2 = cap2.read(frame2), ok3 = cap3.read(frame3);
            if (!ok1 || !ok2 || !ok3) {
                std::cout << "Video ended or read error" << std::endl;
                break;
            }
            ++frameCnt;
            auto t0 = std::chrono::high_resolution_clock::now();
            TripleViewResult result = processor->process_triple_views(frame1, frame2, frame3);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto el = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            totalMs += el;
            /* ---------- Display View Results ---------- */
            if (!result.view1_result.processed_image.empty()) {
                cv::imshow("View1 - 准备/吸屏2", result.view1_result.processed_image);
            }
            if (!result.view2_result.processed_image.empty()) {
                cv::imshow("View2 - 主操作", result.view2_result.processed_image);
            }
            // 显示独立的pose窗口（如果不使用叠加模式）
            if (show_separate_pose_window && processor->get_pose_overlay() &&
                !result.view2_pose_result.processed_image.empty()) {
                cv::imshow("View2 - Pose Only", result.view2_pose_result.processed_image);
            }
            if (!result.view3_result.processed_image.empty()) {
                cv::imshow("View3 - 放回", result.view3_result.processed_image);
            }
            /* ---------- Update Work Sequence Status Display ---------- */
            statusImg.setTo(cv::Scalar(30, 30, 30));
            int y_offset = 30;
            cv::putText(statusImg, "Triple View Work Flow Detection", { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0, 200, 255 }, 2);
            y_offset += 35;
            cv::putText(statusImg, "Frame: " + std::to_string(frameCnt) + " Time: " + std::to_string(el) + "ms",
                { 10, y_offset }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 255, 255, 255 }, 1);
            y_offset += 25;
            cv::putText(statusImg, "Avg Time: " + (frameCnt ? std::to_string(totalMs / frameCnt) : "0") + "ms Cycles: " + std::to_string(result.cycle_count),
                { 10, y_offset }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 255, 255, 255 }, 1);
            y_offset += 25;
            // 显示Pose状态
            std::string pose_status = "Pose: " + std::string(processor->get_pose_overlay() ? "ENABLED" : "DISABLED");
            cv::putText(statusImg, pose_status, { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                processor->get_pose_overlay() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
            y_offset += 35;
            cv::putText(statusImg, "=== CURRENT PHASE ===", { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.7, { 100, 255, 255 }, 2);
            y_offset += 30;
            cv::Scalar phase_color = { 0, 255, 0 };
            if (result.combined_state == ActionState::NG) phase_color = { 0, 0, 255 };
            else if (result.combined_state == ActionState::DETECTED) phase_color = { 0, 255, 255 };
            else if (result.combined_state == ActionState::IDLE) phase_color = { 255, 255, 255 };
            std::string phase_text = processor->get_current_phase_name();
            cv::putText(statusImg, phase_text, { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.9, phase_color, 2);
            y_offset += 25;
            std::string state_text = "State: " + actionStateToString(result.combined_state);
            cv::putText(statusImg, state_text, { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 1);
            y_offset += 40;
            cv::putText(statusImg, "=== WORK SEQUENCE (10 Phases) ===", { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.6, { 200, 200, 100 }, 1);
            y_offset += 25;
            struct PhaseInfo {
                std::string name;
                std::string label;
                WorkPhase phase_enum;
            };
            std::vector<PhaseInfo> phases = {
                {"1.准备", "pad", WorkPhase::PREPARE},
                {"2.吸屏1", "pick", WorkPhase::PICK_SCREEN_1},
                {"3.贴膜", "tape", WorkPhase::APPLY_FILM},
                {"4.撕膜", "tape", WorkPhase::RIP_FILM},
                {"5.按压", "press", WorkPhase::PRESS},
                {"6.棉刷", "cotton", WorkPhase::COTTON_BRUSH},
                {"7.取支架", "zhijia", WorkPhase::TAKE_BRACKET},
                {"8.按压支架", "zhijia", WorkPhase::PRESS_BRACKET},
                {"9.放回", "zhijia", WorkPhase::PUT_BACK},
                {"10.吸屏2", "pick", WorkPhase::PICK_SCREEN_2}
            };
            for (const auto& phase : phases) {
                bool is_current = (result.current_phase == phase.phase_enum);
                bool is_completed = (static_cast<int>(result.current_phase) > static_cast<int>(phase.phase_enum));
                cv::Scalar text_color;
                int thickness = 1;
                if (is_current) {
                    text_color = cv::Scalar(0, 255, 255);
                    thickness = 2;
                    cv::rectangle(statusImg, cv::Point(5, y_offset - 18), cv::Point(200, y_offset + 5),
                        cv::Scalar(100, 100, 50), -1);
                }
                else if (is_completed) {
                    text_color = cv::Scalar(0, 200, 0);
                }
                else {
                    text_color = cv::Scalar(150, 150, 150);
                }
                cv::putText(statusImg, phase.name, { 10, y_offset },
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, text_color, thickness);
                cv::putText(statusImg, "(" + phase.label + ")", { 120, y_offset },
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 180), 1);
                if (&phase != &phases.back()) {
                    cv::putText(statusImg, "v", { 180, y_offset },
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);
                }
                y_offset += 22;
            }
            y_offset += 10;
            cv::putText(statusImg, "=== ACTION STATUS ===", { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.6, { 200, 200, 100 }, 1);
            y_offset += 25;
            std::string current_action = action_manager->get_current_action();
            std::string current_status = action_manager->get_current_status();
            std::string current_reason = action_manager->get_current_reason();
            cv::Scalar status_color = (current_status == "OK") ? cv::Scalar(0, 255, 0) :
                (current_status == "NG") ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
            if (current_action.length() > 40) {
                current_action = current_action.substr(0, 37) + "...";
            }
            cv::putText(statusImg, "Last: " + current_action, { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            y_offset += 22;
            cv::putText(statusImg, "Status: " + current_status, { 10, y_offset },
                cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2);
            if (!current_reason.empty()) {
                y_offset += 22;
                std::string display_reason = current_reason;
                if (display_reason.length() > 45) {
                    display_reason = display_reason.substr(0, 42) + "...";
                }
                cv::putText(statusImg, "Reason: " + display_reason, { 10, y_offset },
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
            cv::imshow("Work Sequence Status", statusImg);
        }
        int key = cv::waitKey(paused ? 0 : 1);
        if (key == 27) break;
        if (key == 32) {
            paused = !paused;
            std::cout << (paused ? "PAUSED" : "RESUMED") << std::endl;
        }
        if (key == 'r' || key == 'R') {
            processor->reset_states();
            frameCnt = 0;
            totalMs = 0;
            std::cout << "Work sequence reset to initial state\n";
        }
        if (key == 'i' || key == 'I') {
            std::cout << "=== System Information ===\n";
            std::cout << "Total frames processed: " << frameCnt << "\n";
            std::cout << "Average processing time: " << (frameCnt ? totalMs / frameCnt : 0) << "ms\n";
            std::cout << "Current cycle count: " << processor->get_cycle_count() << "\n";
            std::cout << "Current phase: " << processor->get_current_phase_name() << "\n";
            std::cout << "Pose overlay: " << (processor->get_pose_overlay() ? "ENABLED" : "DISABLED") << "\n";
        }
        // 新增：P键切换Pose叠加显示
        if (key == 'p' || key == 'P') {
            processor->set_pose_overlay(!processor->get_pose_overlay());
            std::cout << "Pose overlay: " << (processor->get_pose_overlay() ? "ENABLED" : "DISABLED") << std::endl;
        }
    }
    cap1.release();
    cap2.release();
    cap3.release();
    cv::destroyAllWindows();
    std::cout << "Processing completed! Total frames: " << frameCnt
        << ", Average latency: " << (frameCnt ? totalMs / frameCnt : 0) << "ms\n";
    return 0;
}
