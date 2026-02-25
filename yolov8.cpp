#include "yolov8_batch.h"
#include <cassert>
#include <chrono>
#include <set>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <deque>
#include <functional>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"
#include <filesystem>

namespace fs = std::filesystem;

#define CHECK_CUDA_11(call) \
do { \
    const cudaError_t error_code = call; \
    if (error_code != cudaSuccess) { \
        printf("CUDA Error:\n"); \
        printf(" File: %s\n", __FILE__); \
        printf(" Line: %d\n", __LINE__); \
        printf(" Error code: %d\n", error_code); \
        printf(" Error text: %s\n", cudaGetErrorString(error_code)); \
        exit(1); \
    } \
} while (0)

// ------------------- Skeleton Definition (COCO 17 keypoints) -------------------
const std::vector<std::vector<int>> POSE_SKELETON = {
    {5,7}, {7,9}, {6,8}, {8,10}, {5,6}, {5,11},
    {6,12}, {11,12}, {11,13}, {13,15}, {12,14}, {14,16}
}; // COCO keypoints 0-16

// COCO keypoint names
const std::vector<std::string> KEYPOINT_NAMES = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

// ------------------- Colors and Classes -------------------
std::vector<std::string> CLASS_NAMES = {
    "pad", "zhijia", "empty", "pick", "leaf", "tape",
    "cotton", "rip", "press", "ready", "desiccant",
    "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static void try_load_class_names(const std::string& engine_file_path) {
    std::string try1 = engine_file_path + ".names";
    std::string dir;
    size_t pos = engine_file_path.find_last_of("/\\");
    if (pos != std::string::npos) dir = engine_file_path.substr(0, pos + 1);
    std::string try2 = dir + "labels.txt";
    std::vector<std::string> names;
    std::ifstream ifs(try1);
    if (!ifs.is_open()) {
        ifs.open(try2);
    }
    if (!ifs.is_open()) return;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        names.push_back(line);
    }
    if (!names.empty()) {
        CLASS_NAMES = std::move(names);
        std::cerr << "[INFO] Loaded " << CLASS_NAMES.size() << " class names from file." << std::endl;
    }
}

const std::vector<std::vector<unsigned int>> COLORS = {
    { 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
    { 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
    { 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
    { 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
    { 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
    { 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
    { 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
    { 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
    { 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
    { 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
    { 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
    { 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
    { 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
    { 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
    { 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
    { 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
    { 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
    { 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
    { 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
    { 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
    { 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
    { 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
    { 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
    { 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
    { 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
    { 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
    { 80, 183, 189 }, { 128, 128, 0 }
};

constexpr float CONF_THRESH = 0.45f;
constexpr float IOU_THRESH = 0.50f;
constexpr float KPT_THRESH = 0.50f;

/* ====================== Utility Functions and Logger ====================== */
struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};

inline int get_size_by_dims(const nvinfer1::Dims& dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType) {
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 4;
    }
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;
    explicit Logger(nvinfer1::ILogger::Severity severity =
        nvinfer1::ILogger::Severity::kERROR)
        : reportableSeverity(severity) {
    }
    void log(nvinfer1::ILogger::Severity severity,
        const char* msg) noexcept override {
        if (severity > reportableSeverity) return;
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: "; break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: "; break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: "; break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: "; break;
        default:
            std::cerr << "VERBOSE: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

/* ====================== ActionManager Implementation ====================== */
ActionManager::ActionManager() {
    current_action = "Waiting for initial action";
    current_status = "OK";
    current_reason = "";
    last_action_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    // Priority for 10 phases
    action_priority = {
        {"Prepare", 0},
        {"PickScreen1", 1},
        {"ApplyFilm", 2},
        {"RipFilm", 3},
        {"Press", 4},
        {"CottonBrush", 5},
        {"TakeBracket", 6},
        {"PressBracket", 7},
        {"PutBack", 8},
        {"PickScreen2", 9}
    };
}

void ActionManager::update_action(const std::string& new_action, const std::string& source_name,
    const std::string& status, const std::string& reason) {
    if (new_action.empty()) return;
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string current_action_name = current_action;
    size_t pos = current_action.find(" (");
    if (pos != std::string::npos) {
        current_action_name = current_action.substr(0, pos);
    }
    int current_priority = -1;
    auto it_current = action_priority.find(current_action_name);
    if (it_current != action_priority.end()) {
        current_priority = it_current->second;
    }
    int new_priority = -1;
    auto it_new = action_priority.find(new_action);
    if (it_new != action_priority.end()) {
        new_priority = it_new->second;
    }
    bool should_update = false;
    if (new_priority > current_priority) {
        should_update = true;
    }
    else if (new_priority == current_priority &&
        (current_time - last_action_time) > 5000) {
        should_update = true;
    }
    if (should_update) {
        current_action = new_action + " (" + source_name + ")";
        current_status = status;
        current_reason = reason;
        last_action_time = current_time;
        if (action_callback_) {
            action_callback_(current_action, current_status, current_reason);
        }
    }
}

/* ====================== YOLOv8Impl Definition ====================== */
class YOLOv8Impl {
public:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };
    int bs = 1;
    int dtype_size = 4;
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs;
    std::vector<PreParam> pparams;
    std::shared_ptr<ActionManager> action_manager;
    DetectorType detector_type;
    ModelType model_type_; // 模型类型
    YOLOv8Impl(const std::string& engine_file_path, int batch_size, DetectorType detector_type, ModelType model_type);
    ~YOLOv8Impl();
    void make_pipe(bool warmup = true);
    void copyFromMat(const std::vector<cv::Mat>& images, const cv::Size& size);
    void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size);
    void infer();
    void postprocess(std::vector<std::vector<Object>>& objs);
    void postprocess_pose(std::vector<std::vector<PoseObject>>& pose_objs);
    AlgoStatus algo(const std::vector<Object>& objs,
        const std::vector<std::string>& CLASS_NAMES);
    void reset_algo_state();
    void set_detector_type(DetectorType type);
    DetectionResult process_frame_sync(const cv::Mat& frame);
    std::vector<int> nms_cxcywh(const std::vector<float>& boxes,
        const std::vector<float>& scores,
        float iou_threshold);
private:
    void add_detection_info_to_image(DetectionResult& result);
    int step1_entry_count = 0;
    bool prev_step_was_not_1 = true;
};

YOLOv8Impl::YOLOv8Impl(const std::string& engine_file_path, int batch_size, DetectorType detector_type, ModelType model_type)
    : detector_type(detector_type), model_type_(model_type) {
    std::cout << "Loading TensorRT engine: " << engine_file_path << std::endl;
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << engine_file_path << std::endl;
        throw std::runtime_error("Cannot open engine file");
    }
    file.seekg(0, std::ios::end);
    auto sz = file.tellg();
    file.seekg(0, std::ios::beg);
    if (sz == 0) {
        std::cerr << "Error: Engine file is empty" << std::endl;
        throw std::runtime_error("Engine file is empty");
    }
    std::cout << "Engine file size: " << sz << " bytes" << std::endl;
    char* trtModelStream = new char[sz];
    if (!trtModelStream) {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        throw std::runtime_error("Memory allocation failed");
    }
    file.read(trtModelStream, sz);
    file.close();
    try {
        initLibNvInferPlugins(&this->gLogger, "");
        this->runtime = nvinfer1::createInferRuntime(this->gLogger);
        if (!this->runtime) {
            std::cerr << "Error: Failed to create TensorRT Runtime" << std::endl;
            throw std::runtime_error("Failed to create Runtime");
        }
        this->engine = this->runtime->deserializeCudaEngine(trtModelStream, sz);
        if (!this->engine) {
            std::cerr << "Error: Failed to deserialize CUDA engine" << std::endl;
            throw std::runtime_error("Failed to deserialize engine");
        }
        this->context = this->engine->createExecutionContext();
        if (!this->context) {
            std::cerr << "Error: Failed to create execution context" << std::endl;
            throw std::runtime_error("Failed to create execution context");
        }
        cudaError_t stream_status = cudaStreamCreate(&this->stream);
        if (stream_status != cudaSuccess) {
            std::cerr << "Error: Failed to create CUDA stream: " << cudaGetErrorString(stream_status) << std::endl;
            throw std::runtime_error("Failed to create CUDA stream");
        }
        this->num_bindings = this->engine->getNbBindings();
        std::cout << "Number of bindings: " << this->num_bindings << std::endl;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(0);
        dtype_size = type_to_size(dtype);
        for (int i = 0; i < this->num_bindings; ++i) {
            Binding binding;
            binding.name = this->engine->getBindingName(i);
            binding.dsize = type_to_size(this->engine->getBindingDataType(i));
            bool isInput = this->engine->bindingIsInput(i);
            if (isInput) {
                ++this->num_inputs;
                binding.dims = this->engine->getProfileDimensions(
                    i, 0, nvinfer1::OptProfileSelector::kMAX);
                binding.size = get_size_by_dims(binding.dims);
                this->input_bindings.push_back(binding);
                this->context->setBindingDimensions(i, binding.dims);
            }
            else {
                binding.dims = this->context->getBindingDimensions(i);
                binding.size = get_size_by_dims(binding.dims);
                this->output_bindings.push_back(binding);
                ++this->num_outputs;
            }
        }
        std::cout << "Inputs: " << this->num_inputs << ", Outputs: " << this->num_outputs << std::endl;
        std::cout << "TensorRT engine loaded successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "TensorRT initialization exception: " << e.what() << std::endl;
        if (trtModelStream) delete[] trtModelStream;
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
        if (stream) cudaStreamDestroy(stream);
        throw;
    }
    delete[] trtModelStream;
    try_load_class_names(engine_file_path);
    action_manager = std::make_shared<ActionManager>();
    reset_algo_state();
}

YOLOv8Impl::~YOLOv8Impl() {
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    if (stream) cudaStreamDestroy(stream);
    for (auto& p : device_ptrs) if (p) CHECK_CUDA_11(cudaFree(p));
    for (auto& p : host_ptrs) if (p) CHECK_CUDA_11(cudaFreeHost(p));
}

void YOLOv8Impl::reset_algo_state() {
    step1_entry_count = 0;
    prev_step_was_not_1 = true;
}

void YOLOv8Impl::set_detector_type(DetectorType type) {
    detector_type = type;
    reset_algo_state();
}

void YOLOv8Impl::make_pipe(bool warmup) {
    for (auto& b : input_bindings) {
        void* d; CHECK_CUDA_11(cudaMallocAsync(&d, b.size * b.dsize, stream));
        device_ptrs.push_back(d);
    }
    for (auto& b : output_bindings) {
        void* d, * h; size_t sz = b.size * b.dsize;
        CHECK_CUDA_11(cudaMallocAsync(&d, sz, stream));
        CHECK_CUDA_11(cudaHostAlloc(&h, sz, 0));
        device_ptrs.push_back(d);
        host_ptrs.push_back(h);
    }
    if (warmup) {
        for (int i = 0; i < 10; ++i) {
            for (auto& b : input_bindings) {
                size_t sz = b.size * b.dsize;
                void* h = malloc(sz); memset(h, 0, sz);
                CHECK_CUDA_11(cudaMemcpyAsync(device_ptrs[0], h, sz,
                    cudaMemcpyHostToDevice, stream));
                free(h);
            }
            infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8Impl::letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size) {
    const float inp_h = static_cast<float>(size.height);
    const float inp_w = static_cast<float>(size.width);
    float height = static_cast<float>(image.rows);
    float width = static_cast<float>(image.cols);
    float r = std::min(inp_h / height, inp_w / width);
    int padh = static_cast<int>(std::round(height * r));
    int padw = static_cast<int>(std::round(width * r));
    cv::Mat tmp;
    if (static_cast<int>(width) != padw || static_cast<int>(height) != padh)
        cv::resize(image, tmp, cv::Size(padw, padh));
    else
        tmp = image.clone();
    float dw = inp_w - padw, dh = inp_h - padh;
    dw /= 2.0f; dh /= 2.0f;
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0),
        true, false, CV_32F);
    PreParam pparam{ 1 / r, dw, dh, height, width };
    pparams.push_back(pparam);
}

void YOLOv8Impl::copyFromMat(const std::vector<cv::Mat>& images,
    const cv::Size& size) {
    if (device_ptrs.empty()) {
        throw std::runtime_error(
            "[YOLOv8Impl::copyFromMat] device_ptrs not initialized - call make_pipe() first.");
    }
    int real_bs = static_cast<int>(images.size());
    context->setBindingDimensions(
        0, nvinfer1::Dims4{ real_bs, 3, size.height, size.width });
    int total_bytes = real_bs * 3 * size.height * size.width * dtype_size;
    float* temp = (float*)malloc(total_bytes);
    cv::Mat nchw;
    for (int i = 0; i < real_bs; ++i) {
        letterbox(images[i], nchw, size);
        int offset = static_cast<int>(nchw.total()) * i;
        memcpy(temp + offset, nchw.ptr<float>(),
            nchw.total() * nchw.elemSize());
    }
    CHECK_CUDA_11(cudaMemcpyAsync(device_ptrs[0], temp, total_bytes,
        cudaMemcpyHostToDevice, stream));
    free(temp);
}

void YOLOv8Impl::infer() {
    if (device_ptrs.empty()) {
        throw std::runtime_error(
            "[YOLOv8Impl::infer] device_ptrs not initialized - call make_pipe() first.");
    }
    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; ++i) {
        size_t sz = output_bindings[i].size * output_bindings[i].dsize;
        CHECK_CUDA_11(cudaMemcpyAsync(host_ptrs[i],
            device_ptrs[num_inputs + i],
            sz, cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

void YOLOv8Impl::postprocess(std::vector<std::vector<Object>>& batch_objs) {
    batch_objs.clear();
    if (host_ptrs.size() >= 4) {
        if (pparams.size() < static_cast<size_t>(bs)) {
            throw std::runtime_error("[postprocess] pparams.size() < batch size");
        }
        int* num_dets = static_cast<int*>(host_ptrs[0]);
        float* boxes = static_cast<float*>(host_ptrs[1]);
        float* scores = static_cast<float*>(host_ptrs[2]);
        int* labels = static_cast<int*>(host_ptrs[3]);
        if (!num_dets || !boxes || !scores || !labels) {
            throw std::runtime_error("[postprocess] One of output host_ptrs is null");
        }
        for (int b = 0; b < bs; ++b) {
            int n = num_dets[b];
            if (n < 0 || n > 1000) { std::cerr << "[postprocess] suspicious num_dets[" << b << "]=" << n << std::endl; n = 0; }
            auto& pp = pparams[b];
            std::vector<Object> objs;
            for (int i = 0; i < n; ++i) {
                size_t box_idx = static_cast<size_t>(b) * 100 + static_cast<size_t>(i);
                float* bbx = boxes + box_idx * 4;
                float x0 = clamp((bbx[0] - pp.dw) * pp.ratio, 0.f, pp.width);
                float y0 = clamp((bbx[1] - pp.dh) * pp.ratio, 0.f, pp.height);
                float x1 = clamp((bbx[2] - pp.dw) * pp.ratio, 0.f, pp.width);
                float y1 = clamp((bbx[3] - pp.dh) * pp.ratio, 0.f, pp.height);
                Object obj; obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
                obj.prob = scores[static_cast<size_t>(b) * 100 + static_cast<size_t>(i)];
                obj.label = labels[static_cast<size_t>(b) * 100 + static_cast<size_t>(i)];
                objs.push_back(obj);
            }
            batch_objs.push_back(objs);
        }
        return;
    }
    if (host_ptrs.size() == 1 && output_bindings.size() >= 1) {
        auto sigmoidf = [](float x)->float { return 1.0f / (1.0f + std::exp(-x)); };
        float* out = static_cast<float*>(host_ptrs[0]);
        const Binding& Bnd = output_bindings[0];
        if (Bnd.dims.nbDims != 3) {
            throw std::runtime_error("[postprocess] Unexpected fused output nbDims != 3");
        }
        int d0 = Bnd.dims.d[0], d1 = Bnd.dims.d[1], d2 = Bnd.dims.d[2];
        int C = d1, L = d2;
        bool layout_BCL = true;
        if (d1 > 1024 && d2 <= 1024) { layout_BCL = false; C = d2; L = d1; }
        int num_classes_in_model = C - 5;
        int inp_h = 640, inp_w = 640;
        if (!input_bindings.empty() && input_bindings[0].dims.nbDims >= 4) {
            inp_h = input_bindings[0].dims.d[input_bindings[0].dims.nbDims - 2];
            inp_w = input_bindings[0].dims.d[input_bindings[0].dims.nbDims - 1];
        }
        auto at = [&](int bi, int cb, int li)->float {
            if (layout_BCL) return out[bi * (C * L) + cb * L + li];
            else return out[bi * (L * C) + li * C + cb];
            };
        auto iou = [](const Object& a, const Object& b)->float {
            float x1 = std::max(a.rect.x, b.rect.x);
            float y1 = std::max(a.rect.y, b.rect.y);
            float x2 = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
            float y2 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
            float w = std::max(0.f, x2 - x1), h = std::max(0.f, y2 - y1);
            float inter = w * h;
            float ua = a.rect.width * a.rect.height + b.rect.width * b.rect.height - inter;
            return ua <= 0.f ? 0.f : inter / ua;
            };
        auto do_nms = [&](std::vector<Object>& objs, float thr) {
            std::sort(objs.begin(), objs.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
            std::vector<bool> keep(objs.size(), true);
            for (size_t i = 0; i < objs.size(); ++i) if (keep[i]) {
                for (size_t j = i + 1; j < objs.size(); ++j) {
                    if (!keep[j]) continue;
                    if (objs[i].label != objs[j].label) continue;
                    if (iou(objs[i], objs[j]) > thr) keep[j] = false;
                }
            }
            std::vector<Object> outv;
            for (size_t i = 0; i < objs.size(); ++i) if (keep[i]) outv.push_back(objs[i]);
            objs.swap(outv);
            };
        const float conf_threshold = 0.25f;
        const float nms_iou = 0.45f;
        for (int bi = 0; bi < std::min(d0, bs); ++bi) {
            std::vector<Object> objs;
            if (pparams.size() <= static_cast<size_t>(bi)) {
                PreParam pp; pp.ratio = 1.0f; pp.dw = 0.f; pp.dh = 0.f; pp.width = static_cast<float>(inp_w); pp.height = static_cast<float>(inp_h);
                pparams.push_back(pp);
            }
            auto& pp = pparams[bi];
            for (int li = 0; li < L; ++li) {
                float cx = at(bi, 0, li);
                float cy = at(bi, 1, li);
                float w = at(bi, 2, li);
                float h = at(bi, 3, li);
                float raw_obj = at(bi, 4, li);
                float obj_conf = sigmoidf(raw_obj);
                int best_c = -1; float best_score_raw = -1e9f;
                for (int c = 5; c < C; ++c) {
                    float sc_raw = at(bi, c, li);
                    if (sc_raw > best_score_raw) { best_score_raw = sc_raw; best_c = c - 5; }
                }
                float best_score = (best_score_raw >= -0.1f && best_score_raw <= 1.1f) ?
                    best_score_raw : sigmoidf(best_score_raw);
                float conf = obj_conf * best_score;
                if (conf < conf_threshold) continue;
                float model_cx = cx, model_cy = cy, model_w = w, model_h = h;
                if (model_cx <= 1.1f && model_cy <= 1.1f && model_w <= 1.1f && model_h <= 1.1f) {
                    model_cx *= inp_w; model_cy *= inp_h; model_w *= inp_w; model_h *= inp_h;
                }
                float x0 = model_cx - model_w * 0.5f;
                float y0 = model_cy - model_h * 0.5f;
                float x1 = model_cx + model_w * 0.5f;
                float y1 = model_cy + model_h * 0.5f;
                float rx0 = clamp((x0 - pp.dw) * pp.ratio, 0.f, pp.width);
                float ry0 = clamp((y0 - pp.dh) * pp.ratio, 0.f, pp.height);
                float rx1 = clamp((x1 - pp.dw) * pp.ratio, 0.f, pp.width);
                float ry1 = clamp((y1 - pp.dh) * pp.ratio, 0.f, pp.height);
                Object obj;
                obj.rect = cv::Rect_<float>(rx0, ry0, rx1 - rx0, ry1 - ry0);
                obj.prob = conf;
                if (best_c < 0) best_c = 0;
                if (best_c >= static_cast<int>(CLASS_NAMES.size())) best_c = static_cast<int>(CLASS_NAMES.size()) - 1;
                obj.label = best_c;
                objs.push_back(obj);
            }
            do_nms(objs, nms_iou);
            batch_objs.push_back(objs);
        }
        return;
    }
    std::ostringstream oss;
    oss << "[postprocess] Unexpected host_ptrs.size()=" << host_ptrs.size() << "\n";
    throw std::runtime_error(oss.str());
}

std::vector<int> YOLOv8Impl::nms_cxcywh(const std::vector<float>& boxes,
    const std::vector<float>& scores,
    float iou_threshold) {
    int n = boxes.size() / 4;
    if (n == 0) return {};
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    // 按score降序排序
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return scores[i] > scores[j];
        });
    std::vector<float> x1(n), y1(n), x2(n), y2(n), areas(n);
    for (int i = 0; i < n; ++i) {
        int b = order[i] * 4;
        x1[i] = boxes[b + 0] - boxes[b + 2] * 0.5f; // cx-w/2
        y1[i] = boxes[b + 1] - boxes[b + 3] * 0.5f;
        x2[i] = boxes[b + 0] + boxes[b + 2] * 0.5f; // cx+w/2
        y2[i] = boxes[b + 1] + boxes[b + 3] * 0.5f;
        areas[i] = boxes[b + 2] * boxes[b + 3];
    }
    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);
    for (int i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        keep.push_back(order[i]);
        for (int j = i + 1; j < n; ++j) {
            if (suppressed[j]) continue;
            float xx1 = std::max(x1[i], x1[j]);
            float yy1 = std::max(y1[i], y1[j]);
            float xx2 = std::min(x2[i], x2[j]);
            float yy2 = std::min(y2[i], y2[j]);
            float w = std::max(0.f, xx2 - xx1);
            float h = std::max(0.f, yy2 - yy1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[j] - inter);
            if (ovr > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return keep;
}

void YOLOv8Impl::postprocess_pose(std::vector<std::vector<PoseObject>>& batch_pose_objs) {
    batch_pose_objs.clear();
    if (host_ptrs.empty() || output_bindings.empty()) return;

    float* out = static_cast<float*>(host_ptrs[0]);
    const Binding& Bnd = output_bindings[0];

    const int NUM_KEYPOINTS = 17;
    const int COORDS_PER_ANCHOR = 4 + 1 + NUM_KEYPOINTS * 3; // 56
    const int NUM_ANCHORS = 8400;

    auto sigmoidf = [](float x) -> float {
        return 1.0f / (1.0f + std::exp(-std::clamp(x, -50.f, 50.f)));
        };

    for (int bi = 0; bi < std::min(static_cast<int>(Bnd.dims.d[0]), bs); ++bi) {
        std::vector<PoseObject> pose_objs;

        if (pparams.size() <= static_cast<size_t>(bi)) {
            PreParam pp{ 1.0f, 0.f, 0.f, 640.f, 640.f };
            pparams.push_back(pp);
        }
        auto& pp = pparams[bi];

        std::vector<std::vector<float>> pred(NUM_ANCHORS, std::vector<float>(COORDS_PER_ANCHOR));
        for (int a = 0; a < NUM_ANCHORS; ++a) {
            for (int c = 0; c < COORDS_PER_ANCHOR; ++c) {
                pred[a][c] = out[bi * (COORDS_PER_ANCHOR * NUM_ANCHORS) + c * NUM_ANCHORS + a];
            }
        }

        std::vector<int> candidates;
        for (int i = 0; i < NUM_ANCHORS; ++i) {
            float obj_conf = sigmoidf(pred[i][4]);
            if (obj_conf <= POSE_CONF_THRESH) continue;

            float kpt_sum = 0.0f;
            int visible_cnt = 0;
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                float kconf = sigmoidf(pred[i][5 + k * 3 + 2]);
                kpt_sum += kconf;
                if (kconf > POSE_KPT_THRESH) ++visible_cnt;
            }
            float kpt_avg = kpt_sum / NUM_KEYPOINTS;

            if (obj_conf > POSE_CONF_THRESH &&
                kpt_avg > KPT_AVG_MIN_CONF &&
                visible_cnt >= MIN_VISIBLE_KPTS) {
                candidates.push_back(i);
            }
        }

        std::vector<float> boxes, scores;
        for (int idx : candidates) {
            boxes.push_back(pred[idx][0]);
            boxes.push_back(pred[idx][1]);
            boxes.push_back(pred[idx][2]);
            boxes.push_back(pred[idx][3]);
            scores.push_back(sigmoidf(pred[idx][4]));
        }

        std::vector<int> keep = nms_cxcywh(boxes, scores, POSE_IOU_THRESH);

        for (int k_idx : keep) {
            int idx = candidates[k_idx];
            const auto& det = pred[idx];

            float cx = det[0], cy = det[1], w = det[2], h = det[3];
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            float rx1 = clamp((x1 - pp.dw) * pp.ratio, 0.f, pp.width);
            float ry1 = clamp((y1 - pp.dh) * pp.ratio, 0.f, pp.height);
            float rw = (x2 - x1) * pp.ratio;
            float rh = (y2 - y1) * pp.ratio;

            PoseObject obj;
            obj.rect = cv::Rect_<float>(rx1, ry1, rw, rh);
            obj.conf = sigmoidf(det[4]);

            obj.keypoints.clear();
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                int offset = 5 + k * 3;
                float kx = det[offset + 0];
                float ky = det[offset + 1];
                float kconf = sigmoidf(det[offset + 2]);

                float rkx = clamp((kx - pp.dw) * pp.ratio, 0.f, pp.width);
                float rky = clamp((ky - pp.dh) * pp.ratio, 0.f, pp.height);

                Keypoint kp{ rkx, rky, kconf, kconf > POSE_KPT_THRESH };
                obj.keypoints.push_back(kp);
            }
            pose_objs.push_back(obj);
        }

        if (pose_objs.size() > MAX_POSE_OBJECTS) {
            std::sort(pose_objs.begin(), pose_objs.end(),
                [](const PoseObject& a, const PoseObject& b) { return a.conf > b.conf; });
            pose_objs.resize(MAX_POSE_OBJECTS);
        }

        std::cout << "[POSE DEBUG] bi=" << bi
            << " candidates=" << candidates.size()
            << " after_NMS=" << keep.size()
            << " final=" << pose_objs.size() << std::endl;

        batch_pose_objs.push_back(std::move(pose_objs));
    }
}

AlgoStatus YOLOv8Impl::algo(const std::vector<Object>& objs,
    const std::vector<std::string>& CLASS_NAMES) {
    AlgoStatus st;
    st.step = 0;
    st.step_name = "";
    st.status = "idle";
    return st;
}

DetectionResult YOLOv8Impl::process_frame_sync(const cv::Mat& frame) {
    DetectionResult result;
    result.model_type = model_type_;
    pparams.clear();
    std::vector<cv::Mat> batch{ frame };
    cv::Size input_size(640, 640);
    if (!input_bindings.empty() && input_bindings[0].dims.nbDims >= 4) {
        int h = input_bindings[0].dims.d[input_bindings[0].dims.nbDims - 2];
        int w = input_bindings[0].dims.d[input_bindings[0].dims.nbDims - 1];
        if (h > 0 && w > 0) input_size = cv::Size(w, h);
    }
    copyFromMat(batch, input_size);
    infer();
    result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (model_type_ == ModelType::POSE) {
        std::vector<std::vector<PoseObject>> batch_pose_objs;
        postprocess_pose(batch_pose_objs);
        if (!batch_pose_objs.empty()) {
            result.pose_objects = batch_pose_objs[0];
        }
        if (!result.pose_objects.empty()) {
            YOLOv8::draw_pose(frame, result.processed_image, result.pose_objects, POSE_SKELETON);
        }
        else {
            result.processed_image = frame.clone();
            cv::putText(result.processed_image, "No pose detected", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        result.current_action.action = action_manager->get_current_action();
        result.current_action.source_name = "PoseView";
        result.current_action.state = (action_manager->get_current_status() == "OK") ? ActionState::OK : ActionState::NG;
        result.current_action.reason = action_manager->get_current_reason();
        result.current_action.timestamp = result.timestamp;
    }
    else {
        std::vector<std::vector<Object>> batch_objs;
        postprocess(batch_objs);
        if (!batch_objs.empty()) {
            result.detected_objects = batch_objs[0];
        }
        for (const auto& obj : result.detected_objects) {
            if (obj.label < CLASS_NAMES.size()) {
                std::string label_name = CLASS_NAMES[obj.label];
                result.label_count[label_name]++;
            }
        }
        if (!batch_objs.empty()) {
            result.algo_status = algo(batch_objs[0], CLASS_NAMES);
        }
        if (!batch_objs.empty()) {
            YOLOv8::draw_objects(frame, result.processed_image, batch_objs[0], CLASS_NAMES, COLORS);
        }
        else {
            result.processed_image = frame.clone();
        }
        add_detection_info_to_image(result);
        if (result.algo_status.step == 1 && prev_step_was_not_1) {
            ++step1_entry_count;
            prev_step_was_not_1 = false;
        }
        else if (result.algo_status.step != 1) {
            prev_step_was_not_1 = true;
        }
        result.id = step1_entry_count;
        result.current_action.action = action_manager->get_current_action();
        result.current_action.source_name = "Single View";
        result.current_action.state = (action_manager->get_current_status() == "OK") ? ActionState::OK : ActionState::NG;
        result.current_action.reason = action_manager->get_current_reason();
        result.current_action.timestamp = result.timestamp;
    }
    return result;
}

void YOLOv8Impl::add_detection_info_to_image(DetectionResult& result) {
    cv::Mat& image = result.processed_image;
    std::string state_str;
    cv::Scalar state_color;
    switch (result.current_action.state) {
    case ActionState::IDLE: state_str = "IDLE"; state_color = cv::Scalar(255, 255, 255); break;
    case ActionState::DETECTED: state_str = "DETECTED"; state_color = cv::Scalar(255, 255, 0); break;
    case ActionState::OK: state_str = "OK"; state_color = cv::Scalar(0, 255, 0); break;
    case ActionState::NG: state_str = "NG"; state_color = cv::Scalar(0, 0, 255); break;
    }
    std::string type_str = (result.current_action.type == ActionType::INSTANT) ? "INSTANT" : "CONTINUOUS";
    std::string action_text = result.current_action.action + " [" + type_str + " - " + state_str + "]";
    cv::putText(image, action_text, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2);
    std::string time_text = "Timestamp: " + std::to_string(result.timestamp);
    cv::putText(image, time_text, cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    if (!result.current_action.reason.empty()) {
        cv::putText(image, "Reason: " + result.current_action.reason, cv::Point(10, 90),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
    int y_offset = 120;
    for (const auto& [label, count] : result.label_count) {
        std::string count_text = label + ": " + std::to_string(count);
        cv::putText(image, count_text, cv::Point(10, y_offset),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        y_offset += 25;
    }
    if (result.detected_objects.empty()) {
        cv::putText(image, "No objects detected", cv::Point(10, y_offset),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
}

/* ====================== YOLOv8 Wrapper ====================== */
YOLOv8::YOLOv8(const std::string& engine_file_path, int batch_size, DetectorType detector_type, ModelType model_type)
    : impl_(new YOLOv8Impl(engine_file_path, batch_size, detector_type, model_type)),
    detector_type_(detector_type),
    model_type_(model_type) {
    action_manager_ = impl_->action_manager;
}

YOLOv8::~YOLOv8() { delete impl_; }

void YOLOv8::make_pipe(bool warmup) { impl_->make_pipe(warmup); }

void YOLOv8::copyFromMat(const std::vector<cv::Mat>& imgs, const cv::Size& sz) { impl_->copyFromMat(imgs, sz); }

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size) { impl_->letterbox(image, out, size); }

void YOLOv8::infer() { impl_->infer(); }

void YOLOv8::postprocess(std::vector<std::vector<Object>>& objs) { impl_->postprocess(objs); }

void YOLOv8::postprocess_pose(std::vector<std::vector<PoseObject>>& pose_objs) { impl_->postprocess_pose(pose_objs); }

AlgoStatus YOLOv8::algo(const std::vector<Object>& objs, const std::vector<std::string>& CLASS_NAMES) { return impl_->algo(objs, CLASS_NAMES); }

void YOLOv8::reset_algo_state() { impl_->reset_algo_state(); }

void YOLOv8::set_detector_type(DetectorType type) { detector_type_ = type; impl_->set_detector_type(type); }

DetectionResult YOLOv8::process_frame_sync(const cv::Mat& frame) { return impl_->process_frame_sync(frame); }

void YOLOv8::draw_objects(
    const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs,
    const std::vector<std::string>& CLASS_NAMES, const std::vector<std::vector<unsigned int>>& COLORS) {
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        int x = static_cast<int>(obj.rect.x);
        int y = static_cast<int>(obj.rect.y) + 1;
        if (y > res.rows) y = res.rows;
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), cv::Scalar(0, 0, 255), -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
}

void YOLOv8::draw_pose(const cv::Mat& image, cv::Mat& res,
    const std::vector<PoseObject>& pose_objs,
    const std::vector<std::vector<int>>& SKELETON) {
    res = image.clone();
    for (const auto& obj : pose_objs) {
        cv::rectangle(res, obj.rect, cv::Scalar(0, 255, 255), 2);
        cv::putText(res, cv::format("P %.2f", obj.conf),
            cv::Point(obj.rect.x, obj.rect.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        for (size_t i = 0; i < obj.keypoints.size(); ++i) {
            const auto& kpt = obj.keypoints[i];
            if (kpt.visible) {
                cv::circle(res, cv::Point(int(kpt.x), int(kpt.y)), 3, cv::Scalar(0, 255, 0), -1);
            }
        }
        for (const auto& link : SKELETON) {
            int idx1 = link[0];
            int idx2 = link[1];
            if (idx1 < obj.keypoints.size() && idx2 < obj.keypoints.size()) {
                const auto& kpt1 = obj.keypoints[idx1];
                const auto& kpt2 = obj.keypoints[idx2];
                if (kpt1.visible && kpt2.visible) {
                    cv::line(res, cv::Point(int(kpt1.x), int(kpt1.y)),
                        cv::Point(int(kpt2.x), int(kpt2.y)),
                        cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    }
}

void YOLOv8::infer_video(const std::string& video_path, const std::string& out_path, int img_h, int img_w) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return;
    }
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(out_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << out_path << std::endl;
        return;
    }
    cv::Mat frame, res;
    cv::Size img_size(img_w, img_h);
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        impl_->pparams.clear();
        std::vector<cv::Mat> images{ frame };
        impl_->copyFromMat(images, img_size);
        impl_->infer();
        std::vector<std::vector<Object>> batch_objs;
        impl_->postprocess(batch_objs);
        draw_objects(frame, res, batch_objs[0], CLASS_NAMES, COLORS);
        AlgoStatus status = impl_->algo(batch_objs[0], CLASS_NAMES);
        std::string status_text = "Step: " + std::to_string(status.step) + " Status: " + status.status;
        cv::putText(res, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        writer.write(res);
        cv::imshow("YOLOv8 Video", res);
        if (cv::waitKey(1) == 27) break;
    }
    cap.release();
    writer.release();
    cv::destroyAllWindows();
}

void run_inference(const std::string& engine_file_path, const std::vector<std::string>& img_path_list,
    const std::string& out_path, int img_h, int img_w, int batch_size) {
    cudaSetDevice(0);
    auto yolov8 = new YOLOv8(engine_file_path, batch_size);
    yolov8->make_pipe(true);
    cv::Mat image_bgr, res;
    cv::Size img_size(img_w, img_h);
    std::vector<std::vector<Object>> batch_objs;
    std::vector<cv::Mat> images;
    for (const auto& path : img_path_list) {
        image_bgr = cv::imread(path);
        if (image_bgr.empty()) {
            std::cerr << "Failed to read image: " << path << std::endl;
            continue;
        }
        images.push_back(image_bgr);
    }
    batch_objs.clear();
    yolov8->copyFromMat(images, img_size);
    auto start = std::chrono::system_clock::now();
    yolov8->infer();
    auto end = std::chrono::system_clock::now();
    double tc = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "cost " << tc << " ms" << std::endl;
    yolov8->postprocess(batch_objs);
    for (size_t i = 0; i < images.size(); ++i) {
        yolov8->draw_objects(images[i], res, batch_objs[i], CLASS_NAMES, COLORS);
        cv::imwrite(out_path + std::to_string(i + 1) + ".jpg", res);
    }
    delete yolov8;
}

// ====================== Triple View Processor Implementation ======================
TripleViewProcessor::TripleViewProcessor(const std::string& model_path1,
    const std::string& model_path2, const std::string& model_path3,
    const std::string& pose_model_path)
    : pose_model_path_(pose_model_path) {
    detector1 = std::make_unique<YOLOv8>(model_path1, 1, DetectorType::VIDEO_WORKER1, ModelType::DETECTOR);
    detector2 = std::make_unique<YOLOv8>(model_path2, 1, DetectorType::VIDEO_WORKER2, ModelType::DETECTOR);
    detector3 = std::make_unique<YOLOv8>(model_path3, 1, DetectorType::VIDEO_WORKER3, ModelType::DETECTOR);
    if (!pose_model_path.empty() && std::filesystem::exists(pose_model_path)) {
        try {
            detector2_pose = std::make_unique<YOLOv8>(pose_model_path, 1, DetectorType::VIDEO_WORKER2_POSE, ModelType::POSE);
            has_pose_detector_ = true;
            std::cout << "[INFO] Pose detector initialized: " << pose_model_path << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to load pose model: " << e.what() << std::endl;
            has_pose_detector_ = false;
        }
    }
    else {
        std::cout << "[INFO] No pose model specified or file not found, pose detection disabled" << std::endl;
        has_pose_detector_ = false;
    }
    action_manager_ = std::make_shared<ActionManager>();
    phase_names_ = {
        {WorkPhase::PREPARE, "准备"},
        {WorkPhase::PICK_SCREEN_1, "吸屏1"},
        {WorkPhase::APPLY_FILM, "贴膜"},
        {WorkPhase::RIP_FILM, "撕膜"},
        {WorkPhase::PRESS, "按压"},
        {WorkPhase::COTTON_BRUSH, "棉刷"},
        {WorkPhase::TAKE_BRACKET, "取支架"},
        {WorkPhase::PRESS_BRACKET, "按压支架"},
        {WorkPhase::PUT_BACK, "放回"},
        {WorkPhase::PICK_SCREEN_2, "吸屏2"}
    };
    reset_states();
}

void TripleViewProcessor::init() {
    detector1->make_pipe(true);
    detector2->make_pipe(true);
    detector3->make_pipe(true);
    if (has_pose_detector_ && detector2_pose) {
        detector2_pose->make_pipe(true);
        std::cout << "[INFO] Pose detector warmed up successfully" << std::endl;
    }
}

void TripleViewProcessor::reset_states() {
    current_work_phase_ = WorkPhase::PREPARE;
    prepare_buffer_.clear();
    pick_screen_buffer_1_.clear();
    apply_film_buffer_.clear();
    rip_film_buffer_.clear();
    press_buffer_.clear();
    cotton_brush_buffer_.clear();
    take_bracket_buffer_.clear();
    press_bracket_buffer_.clear();
    put_back_buffer_.clear();
    pick_screen_buffer_2_.clear();
    take_bracket_complete_time_ = 0;
    waiting_for_press_bracket_ = false;
    film_was_detected_ = false;
    cycle_count_ = 0;
    last_phase_transition_time_ = 0;
    frame_counter_ = 0;
}

std::string TripleViewProcessor::get_current_phase_name() const {
    auto it = phase_names_.find(current_work_phase_);
    if (it != phase_names_.end()) return it->second;
    return "Unknown";
}

bool TripleViewProcessor::check_label_exists(const DetectionResult& result,
    const std::string& label_name, float confidence_threshold) {
    for (const auto& obj : result.detected_objects) {
        if (obj.label < CLASS_NAMES.size() &&
            CLASS_NAMES[obj.label] == label_name &&
            obj.prob >= confidence_threshold) {
            return true;
        }
    }
    return false;
}

int TripleViewProcessor::count_label(const DetectionResult& result,
    const std::string& label_name, float confidence_threshold) {
    int count = 0;
    for (const auto& obj : result.detected_objects) {
        if (obj.label < CLASS_NAMES.size() &&
            CLASS_NAMES[obj.label] == label_name &&
            obj.prob >= confidence_threshold) {
            count++;
        }
    }
    return count;
}

bool TripleViewProcessor::detect_prepare(const DetectionResult& view1_result) {
    bool has_pad = check_label_exists(view1_result, "pad", 0.5f);
    prepare_buffer_.push_back(has_pad);
    if (prepare_buffer_.size() > 5) prepare_buffer_.pop_front();
    if (prepare_buffer_.size() == 5 &&
        std::all_of(prepare_buffer_.begin(), prepare_buffer_.end(), [](bool v) { return v; })) {
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_pick_screen(const DetectionResult& view2_result) {
    bool has_pick = check_label_exists(view2_result, "pick", 0.5f);
    pick_screen_buffer_1_.push_back(has_pick);
    if (pick_screen_buffer_1_.size() > 5) pick_screen_buffer_1_.pop_front();
    if (pick_screen_buffer_1_.size() == 5 &&
        std::all_of(pick_screen_buffer_1_.begin(), pick_screen_buffer_1_.end(), [](bool v) { return v; })) {
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_apply_film(const DetectionResult& view2_result) {
    bool has_tape = check_label_exists(view2_result, "tape", 0.5f);
    apply_film_buffer_.push_back(has_tape);
    if (apply_film_buffer_.size() > 5) apply_film_buffer_.pop_front();
    if (apply_film_buffer_.size() == 5 &&
        std::all_of(apply_film_buffer_.begin(), apply_film_buffer_.end(), [](bool v) { return v; })) {
        film_was_detected_ = true;
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_rip_film(const DetectionResult& view2_result) {
    bool has_tape = check_label_exists(view2_result, "tape", 0.5f);
    if (film_was_detected_ && !has_tape) {
        rip_film_buffer_.push_back(true);
    }
    else {
        rip_film_buffer_.push_back(false);
    }
    if (rip_film_buffer_.size() > 5) rip_film_buffer_.pop_front();
    if (rip_film_buffer_.size() == 5 &&
        std::all_of(rip_film_buffer_.begin(), rip_film_buffer_.end(), [](bool v) { return v; })) {
        film_was_detected_ = false;
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_press(const DetectionResult& view2_result) {
    bool has_press = check_label_exists(view2_result, "press", 0.5f);
    press_buffer_.push_back(has_press);
    if (press_buffer_.size() > 5) press_buffer_.pop_front();
    if (press_buffer_.size() == 5 &&
        std::all_of(press_buffer_.begin(), press_buffer_.end(), [](bool v) { return v; })) {
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_cotton_brush(const DetectionResult& view2_result) {
    bool has_cotton = check_label_exists(view2_result, "cotton", 0.5f);
    cotton_brush_buffer_.push_back(has_cotton);
    if (cotton_brush_buffer_.size() > 10) cotton_brush_buffer_.pop_front();
    if (cotton_brush_buffer_.size() == 10 &&
        std::all_of(cotton_brush_buffer_.begin(), cotton_brush_buffer_.end(), [](bool v) { return v; })) {
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_take_bracket(const DetectionResult& view2_result) {
    bool has_zhijia = check_label_exists(view2_result, "zhijia", 0.5f);
    take_bracket_buffer_.push_back(has_zhijia);
    if (take_bracket_buffer_.size() > 5) take_bracket_buffer_.pop_front();
    if (take_bracket_buffer_.size() == 5 &&
        std::all_of(take_bracket_buffer_.begin(), take_bracket_buffer_.end(), [](bool v) { return v; })) {
        take_bracket_complete_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        waiting_for_press_bracket_ = true;
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_press_bracket(const DetectionResult& view2_result) {
    if (!waiting_for_press_bracket_) return false;
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if ((current_time - take_bracket_complete_time_) < 300) {
        return false;
    }
    bool has_zhijia = check_label_exists(view2_result, "zhijia", 0.5f);
    press_bracket_buffer_.push_back(has_zhijia);
    if (press_bracket_buffer_.size() > 5) press_bracket_buffer_.pop_front();
    if (press_bracket_buffer_.size() == 5 &&
        std::all_of(press_bracket_buffer_.begin(), press_bracket_buffer_.end(), [](bool v) { return v; })) {
        waiting_for_press_bracket_ = false;
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_put_back(const DetectionResult& view3_result) {
    bool has_zhijia = check_label_exists(view3_result, "zhijia", 0.5f);
    put_back_buffer_.push_back(has_zhijia);
    if (put_back_buffer_.size() > 5) put_back_buffer_.pop_front();
    if (put_back_buffer_.size() == 5 &&
        std::all_of(put_back_buffer_.begin(), put_back_buffer_.end(), [](bool v) { return v; })) {
        return true;
    }
    return false;
}

bool TripleViewProcessor::detect_pick_screen_view1(const DetectionResult& view1_result) {
    bool has_pick = check_label_exists(view1_result, "pick", 0.5f);
    pick_screen_buffer_2_.push_back(has_pick);
    if (pick_screen_buffer_2_.size() > 5) pick_screen_buffer_2_.pop_front();
    if (pick_screen_buffer_2_.size() == 5 &&
        std::all_of(pick_screen_buffer_2_.begin(), pick_screen_buffer_2_.end(), [](bool v) { return v; })) {
        return true;
    }
    return false;
}

TripleViewResult TripleViewProcessor::process_triple_views(
    const cv::Mat& view1, const cv::Mat& view2, const cv::Mat& view3) {
    TripleViewResult result;
    frame_counter_++;
    result.combined_state = ActionState::IDLE;
    result.combined_type = ActionType::CONTINUOUS;
    result.combined_reason = "等待动作开始";
    result.current_phase = current_work_phase_;
    result.cycle_count = cycle_count_;
    result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    result.view1_result = detector1->process_frame_sync(view1);
    result.view2_result = detector2->process_frame_sync(view2);
    result.view3_result = detector3->process_frame_sync(view3);
    if (has_pose_detector_ && detector2_pose) {
        result.view2_pose_result = detector2_pose->process_frame_sync(view2);
        std::cout << "[POSE DEBUG] 這幀偵測到 "
            << result.view2_pose_result.pose_objects.size()
            << " 個人體" << std::endl;
        if (enable_pose_overlay_ && !result.view2_pose_result.processed_image.empty()) {
            cv::Mat overlay;
            cv::Mat poseImg = result.view2_pose_result.processed_image;
            if (result.view2_result.processed_image.empty()) {
                result.view2_result.processed_image = poseImg.clone();
            }
            else {
                if (poseImg.size() != result.view2_result.processed_image.size()) {
                    cv::resize(poseImg, poseImg, result.view2_result.processed_image.size());
                }
                cv::addWeighted(result.view2_result.processed_image, 0.7,
                    poseImg, 0.3, 0, overlay);
                result.view2_result.processed_image = overlay;
            }
        }
    }
    static int global_id = 0;
    result.id = global_id++;
    result.view1_result.id = result.id;
    result.view2_result.id = result.id;
    result.view3_result.id = result.id;
    result.view1_result.timestamp = result.timestamp;
    result.view2_result.timestamp = result.timestamp;
    result.view3_result.timestamp = result.timestamp;
    bool phase_completed = false;
    std::string completed_action_name;
    switch (current_work_phase_) {
    case WorkPhase::PREPARE: {
        result.combined_action = "准备";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测准备阶段...";
        if (detect_prepare(result.view1_result)) {
            phase_completed = true;
            completed_action_name = "Prepare";
            current_work_phase_ = WorkPhase::PICK_SCREEN_1;
            prepare_buffer_.clear();
            std::cout << "=== 准备 完成! 进入吸屏1 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::PICK_SCREEN_1: {
        result.combined_action = "吸屏1";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测吸屏动作(View2)...";
        if (detect_pick_screen(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "PickScreen1";
            current_work_phase_ = WorkPhase::APPLY_FILM;
            pick_screen_buffer_1_.clear();
            std::cout << "=== 吸屏1 完成! 进入贴膜 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::APPLY_FILM: {
        result.combined_action = "贴膜";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测贴膜动作...";
        if (detect_apply_film(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "ApplyFilm";
            current_work_phase_ = WorkPhase::RIP_FILM;
            apply_film_buffer_.clear();
            std::cout << "=== 贴膜 完成! 进入撕膜 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::RIP_FILM: {
        result.combined_action = "撕膜";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测撕膜动作(tape消失)...";
        if (detect_rip_film(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "RipFilm";
            current_work_phase_ = WorkPhase::PRESS;
            rip_film_buffer_.clear();
            std::cout << "=== 撕膜 完成! 进入按压 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::PRESS: {
        result.combined_action = "按压";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测按压动作...";
        if (detect_press(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "Press";
            current_work_phase_ = WorkPhase::COTTON_BRUSH;
            press_buffer_.clear();
            std::cout << "=== 按压 完成! 进入棉刷 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::COTTON_BRUSH: {
        result.combined_action = "棉刷";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测棉刷动作...";
        if (detect_cotton_brush(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "CottonBrush";
            current_work_phase_ = WorkPhase::TAKE_BRACKET;
            cotton_brush_buffer_.clear();
            std::cout << "=== 棉刷 完成! 进入取支架 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::TAKE_BRACKET: {
        result.combined_action = "取支架";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测取支架动作...";
        if (detect_take_bracket(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "TakeBracket";
            current_work_phase_ = WorkPhase::PRESS_BRACKET;
            take_bracket_buffer_.clear();
            std::cout << "=== 取支架 完成! 等待按压支架 (10帧后) ===" << std::endl;
        }
        break;
    }
    case WorkPhase::PRESS_BRACKET: {
        result.combined_action = "按压支架";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测按压支架动作(取支架后10帧+)...";
        if (detect_press_bracket(result.view2_result)) {
            phase_completed = true;
            completed_action_name = "PressBracket";
            current_work_phase_ = WorkPhase::PUT_BACK;
            press_bracket_buffer_.clear();
            std::cout << "=== 按压支架 完成! 进入放回 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::PUT_BACK: {
        result.combined_action = "放回";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测放回动作(View3)...";
        if (detect_put_back(result.view3_result)) {
            phase_completed = true;
            completed_action_name = "PutBack";
            current_work_phase_ = WorkPhase::PICK_SCREEN_2;
            put_back_buffer_.clear();
            std::cout << "=== 放回 完成! 进入吸屏2 ===" << std::endl;
        }
        break;
    }
    case WorkPhase::PICK_SCREEN_2: {
        result.combined_action = "吸屏2";
        result.combined_state = ActionState::DETECTED;
        result.combined_reason = "检测吸屏动作(View1)...";
        if (detect_pick_screen_view1(result.view1_result)) {
            phase_completed = true;
            completed_action_name = "PickScreen2";
            cycle_count_++;
            current_work_phase_ = WorkPhase::PREPARE;
            pick_screen_buffer_2_.clear();
            reset_states();
            current_work_phase_ = WorkPhase::PREPARE;
            std::cout << "=== 吸屏2 完成! 循环结束，回到准备阶段 ===" << std::endl;
            std::cout << "=== 完成第 " << cycle_count_ << " 个循环 ===" << std::endl;
        }
        break;
    }
    }
    if (phase_completed) {
        result.combined_state = ActionState::OK;
        result.combined_reason = completed_action_name + " 动作完成";
        action_manager_->update_action(completed_action_name, "TripleView", "OK", "");
        last_phase_transition_time_ = result.timestamp;
    }
    result.combined_action = get_current_phase_name() + " | 循环: " + std::to_string(cycle_count_);
    return result;
}

/* ====================== C Interface ====================== */
extern "C" {
    void* YOLOv8_Create(const char* engine_file_path, int batch_size) {
        try { return new YOLOv8(std::string(engine_file_path), batch_size); }
        catch (...) { return nullptr; }
    }
    void YOLOv8_Destroy(void* instance) { if (instance) delete static_cast<YOLOv8*>(instance); }
    void YOLOv8_MakePipe(void* instance, bool warmup) { if (instance) static_cast<YOLOv8*>(instance)->make_pipe(warmup); }
    int YOLOv8_GetAlgoStatus(void* instance, void* objs, int num_objs, const char** class_names, int num_classes) { return 0; }
    void YOLOv8_ResetAlgoState(void* instance) { if (instance) static_cast<YOLOv8*>(instance)->reset_algo_state(); }
}
