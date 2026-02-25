#pragma once
#ifndef YOLOV8_H
#define YOLOV8_H

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <deque>
#include <chrono>

// ------------------- DLL Export Macros -------------------
#ifdef YOLOV8_BUILD_DLL
#define YOLOV8_API __declspec(dllexport)
#else
#define YOLOV8_API __declspec(dllimport)
#endif

// ------------------- Forward Declarations -------------------
class YOLOv8Impl;

// ------------------- Model Type Enum -------------------
enum class ModelType {
    DETECTOR = 0, // 标准检测模型 (bbox + class)
    POSE = 1      // 姿态估计模型 (bbox + keypoints)
};

// ------------------- Action Type and State Enums -------------------
enum class ActionType {
    INSTANT = 0,
    CONTINUOUS = 1
};

enum class ActionState {
    IDLE = 0,
    DETECTED = 1,
    OK = 2,
    NG = 3
};

// ------------------- Work Phase Enum (10 Phases) -------------------
enum class WorkPhase {
    PREPARE = 0,
    PICK_SCREEN_1 = 1,
    APPLY_FILM = 2,
    RIP_FILM = 3,
    PRESS = 4,
    COTTON_BRUSH = 5,
    TAKE_BRACKET = 6,
    PRESS_BRACKET = 7,
    PUT_BACK = 8,
    PICK_SCREEN_2 = 9
};

// ------------------- Data Structures -------------------
struct AlgoStatus {
    int step = 0;
    std::string step_name = "";
    std::string status = "idle";
};

struct Object {
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0f;
};

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float conf = 0.0f;
    bool visible = false;
};

struct PoseObject {
    cv::Rect_<float> rect;
    float conf = 0.0f;
    std::vector<Keypoint> keypoints;
};

struct PreParam {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0.0f;
    float width = 0.0f;
};

struct ActionDetection {
    std::string action;
    std::string source_name;
    ActionType type;
    ActionState state;
    std::string reason;
    int64_t detect_time;
    int64_t judge_time;
    int64_t timestamp;
};

struct DetectionResult {
    cv::Mat processed_image;
    AlgoStatus algo_status;
    std::map<std::string, int> label_count;
    std::vector<Object> detected_objects;
    std::vector<PoseObject> pose_objects;
    int64_t timestamp;
    int id;
    ActionDetection current_action;
    ModelType model_type = ModelType::DETECTOR;
};

struct TripleViewResult {
    DetectionResult view1_result;
    DetectionResult view2_result;
    DetectionResult view2_pose_result;
    DetectionResult view3_result;
    std::string combined_action;
    ActionState combined_state;
    std::string combined_reason;
    ActionType combined_type;
    WorkPhase current_phase;
    int64_t timestamp;
    int id;
    int cycle_count;
};

// ------------------- Action Manager -------------------
class YOLOV8_API ActionManager {
public:
    ActionManager();
    void update_action(const std::string& new_action, const std::string& source_name,
        const std::string& status = "OK", const std::string& reason = "");
    std::string get_current_action() const { return current_action; }
    std::string get_current_status() const { return current_status; }
    std::string get_current_reason() const { return current_reason; }
    void set_action_callback(std::function<void(const std::string&, const std::string&, const std::string&)> callback) {
        action_callback_ = callback;
    }
private:
    std::string current_action;
    std::string current_status;
    std::string current_reason;
    int64_t last_action_time;
    std::map<std::string, int> action_priority;
    std::function<void(const std::string&, const std::string&, const std::string&)> action_callback_;
};

// ------------------- Detector Types -------------------
enum class DetectorType {
    VIDEO_WORKER1 = 0,
    VIDEO_WORKER2 = 1,
    VIDEO_WORKER2_POSE = 2,
    VIDEO_WORKER3 = 3
};

// ------------------- Global Constants & Data -------------------
extern std::vector<std::string> CLASS_NAMES;
extern const std::vector<std::vector<unsigned int>> COLORS;
extern const std::vector<std::vector<int>> POSE_SKELETON;

// ------------------- Pose Postprocess Thresholds (最新工业级阈值) -------------------
constexpr float POSE_CONF_THRESH = 0.58f;   // 目标置信度
constexpr float POSE_IOU_THRESH = 0.65f;   // NMS IoU
constexpr float POSE_KPT_THRESH = 0.45f;   // 单点可见性阈值
constexpr float KPT_AVG_MIN_CONF = 0.35f;   // 关键点平均置信度
constexpr int   MIN_VISIBLE_KPTS = 6;       // 至少需要6个可见关键点
constexpr int   MAX_POSE_OBJECTS = 30;      // 硬上限，防止爆炸

// ------------------- YOLOv8 Class -------------------
class YOLOV8_API YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path, int batch_size = 1,
        DetectorType detector_type = DetectorType::VIDEO_WORKER1,
        ModelType model_type = ModelType::DETECTOR);
    ~YOLOv8();
    void make_pipe(bool warmup = true);
    void copyFromMat(const std::vector<cv::Mat>&, const cv::Size&);
    void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size);
    void infer();
    void postprocess(std::vector<std::vector<Object>>& objs);
    void postprocess_pose(std::vector<std::vector<PoseObject>>& pose_objs);
    static void draw_objects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs,
        const std::vector<std::string>& class_names,
        const std::vector<std::vector<unsigned int>>& colors);
    static void draw_pose(const cv::Mat& image, cv::Mat& res, const std::vector<PoseObject>& pose_objs,
        const std::vector<std::vector<int>>& skeleton = POSE_SKELETON);
    void infer_video(const std::string& video_path, const std::string& out_path, int img_h = 640, int img_w = 640);
    AlgoStatus algo(const std::vector<Object>& objs, const std::vector<std::string>& class_names);
    void reset_algo_state();
    void set_detector_type(DetectorType type);
    DetectionResult process_frame_sync(const cv::Mat& frame);
    std::shared_ptr<ActionManager> get_action_manager() { return action_manager_; }
    DetectorType get_detector_type() const { return detector_type_; }
    ModelType get_model_type() const { return model_type_; }
private:
    YOLOv8Impl* impl_;
    std::shared_ptr<ActionManager> action_manager_;
    DetectorType detector_type_;
    ModelType model_type_;
};

// ------------------- Triple View Processor -------------------
class YOLOV8_API TripleViewProcessor {
public:
    TripleViewProcessor(const std::string& model_path1,
        const std::string& model_path2,
        const std::string& model_path3,
        const std::string& pose_model_path = "");
    void init();
    TripleViewResult process_triple_views(const cv::Mat& view1,
        const cv::Mat& view2,
        const cv::Mat& view3);
    std::shared_ptr<ActionManager> get_action_manager() { return action_manager_; }
    void reset_states();
    std::string get_current_phase_name() const;
    int get_cycle_count() const { return cycle_count_; }
    void set_pose_overlay(bool enable) { enable_pose_overlay_ = enable; }
    bool get_pose_overlay() const { return enable_pose_overlay_; }
private:
    bool check_label_exists(const DetectionResult& result,
        const std::string& label_name,
        float confidence_threshold = 0.5f);
    int count_label(const DetectionResult& result,
        const std::string& label_name,
        float confidence_threshold = 0.5f);
    bool detect_prepare(const DetectionResult& view1_result);
    bool detect_pick_screen(const DetectionResult& view2_result);
    bool detect_apply_film(const DetectionResult& view2_result);
    bool detect_rip_film(const DetectionResult& view2_result);
    bool detect_press(const DetectionResult& view2_result);
    bool detect_cotton_brush(const DetectionResult& view2_result);
    bool detect_take_bracket(const DetectionResult& view2_result);
    bool detect_press_bracket(const DetectionResult& view2_result);
    bool detect_put_back(const DetectionResult& view3_result);
    bool detect_pick_screen_view1(const DetectionResult& view1_result);

    std::unique_ptr<YOLOv8> detector1;
    std::unique_ptr<YOLOv8> detector2;
    std::unique_ptr<YOLOv8> detector2_pose;
    std::unique_ptr<YOLOv8> detector3;
    std::shared_ptr<ActionManager> action_manager_;
    WorkPhase current_work_phase_;
    std::map<WorkPhase, std::string> phase_names_;
    std::deque<bool> prepare_buffer_;
    std::deque<bool> pick_screen_buffer_1_;
    std::deque<bool> apply_film_buffer_;
    std::deque<bool> rip_film_buffer_;
    std::deque<bool> press_buffer_;
    std::deque<bool> cotton_brush_buffer_;
    std::deque<bool> take_bracket_buffer_;
    std::deque<bool> press_bracket_buffer_;
    std::deque<bool> put_back_buffer_;
    std::deque<bool> pick_screen_buffer_2_;
    int64_t take_bracket_complete_time_ = 0;
    bool waiting_for_press_bracket_ = false;
    bool film_was_detected_ = false;
    int cycle_count_ = 0;
    int64_t last_phase_transition_time_ = 0;
    int frame_counter_ = 0;
    bool enable_pose_overlay_ = true;
    std::string pose_model_path_;
    bool has_pose_detector_ = false;
};

// ------------------- Global Functions -------------------
YOLOV8_API void run_inference(
    const std::string& engine_file_path,
    const std::vector<std::string>& img_path_list,
    const std::string& out_path,
    int img_h = 640,
    int img_w = 640,
    int batch_size = 4
);

// ------------------- C Interface -------------------
extern "C" {
    YOLOV8_API void* YOLOv8_Create(const char* engine_file_path, int batch_size);
    YOLOV8_API void YOLOv8_Destroy(void* instance);
    YOLOV8_API void YOLOv8_MakePipe(void* instance, bool warmup);
    YOLOV8_API int YOLOv8_GetAlgoStatus(void* instance, void* objs, int num_objs,
        const char** class_names, int num_classes);
    YOLOV8_API void YOLOv8_ResetAlgoState(void* instance);
}

// ------------------- External Frame Processor -------------------
class YOLOV8_API ExternalFrameProcessor {
private:
    std::unique_ptr<YOLOv8> detector;
public:
    ExternalFrameProcessor(const std::string& model_path,
        DetectorType detector_type = DetectorType::VIDEO_WORKER1);
    ExternalFrameProcessor(const ExternalFrameProcessor&) = delete;
    ExternalFrameProcessor& operator=(const ExternalFrameProcessor&) = delete;
    ExternalFrameProcessor(ExternalFrameProcessor&&) = default;
    ExternalFrameProcessor& operator=(ExternalFrameProcessor&&) = default;
    DetectionResult process_frame(const cv::Mat& input_frame);
    std::shared_ptr<ActionManager> get_action_manager();
    YOLOv8* get_detector();
};

#endif // YOLOV8_H
