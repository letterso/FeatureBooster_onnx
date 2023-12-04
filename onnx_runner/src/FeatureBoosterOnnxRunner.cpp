/**
 * @file main.cpp
 * @author letterso
 * @brief
 * @version 0.5
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "FeatureBoosterOnnxRunner.h"

int FeatureBoosterOnnxRunner::InitOrtEnv(const std::string &onnx_file, const unsigned int &env_device)
{
    try
    {
        // init onnxruntime env
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FeatureBoosterOnnxRunner");
        env_device_ = env_device;

        // set options
        msessionOptions = Ort::SessionOptions();
        // msessionOptions.SetIntraOpNumThreads(std::min(6, (int) std::thread::hardware_concurrency()));
        msessionOptions.SetIntraOpNumThreads(6);
        msessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (env_device_ == 1)
        {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;     // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            msessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            msessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

        // 加载模型及或者输入输出
        session = std::make_unique<Ort::Session>(env, onnx_file.c_str(), msessionOptions);
        const size_t numInputNodes = session->GetInputCount();
        InputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            InputNodeNames.emplace_back(strdup(session->GetInputNameAllocated(i, allocator).get()));
            InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        const size_t numOutputNodes = session->GetOutputCount();
        OutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            OutputNodeNames.emplace_back(strdup(session->GetOutputNameAllocated(i, allocator).get()));
            OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int FeatureBoosterOnnxRunner::PreProcess(const cv::Size &imageSize, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &desc, std::vector<std::array<float, 4>> &kpts, std::vector<std::array<float, 256>> &desc_pre)
{
    try
    {
        // 特征点前处理
        float x0 = imageSize.width / 2.0;
        float y0 = imageSize.height / 2.0;
        float scale = (imageSize.width > imageSize.height) ? imageSize.width * 0.7 : imageSize.height * 0.7;
        kpts.reserve(keypoints.size());
        for (cv::KeyPoint kpt : keypoints)
        {
            std::array<float, 4> kpt_pre;
            // 归一化坐标
            kpt_pre[0] = (kpt.pt.x - x0) / scale;
            kpt_pre[1] = (kpt.pt.y - y0) / scale;
            kpt_pre[2] = kpt.size / 31.0;
            kpt_pre[3] = kpt.angle / 180.0 * M_PI;
            kpts.push_back(kpt_pre);
        }

        // 描述子的前处理
        desc_pre.reserve(desc.size().height);
        for (int i = 0; i < desc.size().height; i++)
        {
            std::array<float, 256> desc_pack = unpackbitsLittleEndian(desc.row(i));
            for (unsigned i = 0; i < 256; i++)
                desc_pack[i] = desc_pack[i] * 2.0 - 1.0;
            desc_pre.push_back(desc_pack);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] FeatureBoosterOnnxRunner PreProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int FeatureBoosterOnnxRunner::Inference(const std::vector<std::array<float, 4>> &kpts, const std::vector<std::array<float, 256>> &desc_pre)
{
    try
    {
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        float *desc0_data = new float[desc_pre.size() * 256];
        std::memcpy(desc0_data, desc_pre.data(), desc_pre.size() * 256 * sizeof(float));
        InputNodeShapes[0][0] = desc_pre.size(); // 0维度为动态，需手动指定维度否则为-1

        float *kpts_data = new float[kpts.size() * 4];
        std::memcpy(kpts_data, kpts.data(), kpts.size() * 4 * sizeof(float));
        InputNodeShapes[1][0] = kpts.size(); // 0维度为动态，需手动指定维度否则为-1

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, desc0_data, desc_pre.size() * 256 * sizeof(float),
            InputNodeShapes[0].data(), InputNodeShapes[0].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, kpts_data, kpts.size() * 4 * sizeof(float),
            InputNodeShapes[1].data(), InputNodeShapes[1].size()));

        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, InputNodeNames.data(), input_tensors.data(),
                                          input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());

        for (auto &tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        minferOutputtensors = std::move(output_tensor);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] FeatureBoosterOnnxRunner inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int FeatureBoosterOnnxRunner::PostProcess(cv::Mat &desc_boost)
{
    try
    {
        // load date from tensor
        std::vector<int64_t> desc_Shape = minferOutputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
        float *desc = (float *)minferOutputtensors[0].GetTensorMutableData<void>();

        // Process desc
        desc_boost = cv::Mat(desc_Shape[0], 32, CV_8UC1);
        for (int i = 0; i < desc_Shape[0]; i++)
        {
            std::array<float, 256> in_data;
            std::copy(desc + i * 256, desc + (i + 1) * 256, in_data.begin());
            std::array<uint8_t, 32> pack_data = packbitsLittleEndian(in_data);
            std::memcpy(desc_boost.ptr<uint8_t>(i), pack_data.data(), 32 * sizeof(uint8_t));
        }

        // std::cout << "[INFO] Postprocessing operation completed successfully" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] FeatureBoosterOnnxRunner PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int FeatureBoosterOnnxRunner::InferenceDescriptor(const cv::Size &imageSize, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &desc, cv::Mat &descBoost)
{
    if (keypoints.empty() || desc.empty())
    {
        std::cout << "[ERROR] Keypoints or Descriptor is empty" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::array<float, 4>> kpts;
    std::vector<std::array<float, 256>> desc_pre;
    auto time_start = std::chrono::high_resolution_clock::now();
    PreProcess(imageSize, keypoints, desc, kpts, desc_pre);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "PreProcess cost time : " << diff << std::endl;

    Inference(kpts, desc_pre);
    time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Inference cost time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << std::endl;

    PostProcess(descBoost);
    time_end = std::chrono::high_resolution_clock::now();
    std::cout << "PostProcess cost time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << std::endl;

    return EXIT_SUCCESS;
}

std::array<float, 256> FeatureBoosterOnnxRunner::unpackbitsLittleEndian(const cv::Mat &inData)
{
    std::array<float, 256> result;
    for (size_t byte = 0; byte < 32; ++byte)
    {
        size_t startIdx = byte * 8;
        uint8_t byteValue = inData.at<uint8_t>(0, byte);

        for (size_t bit = 0; bit < 8; ++bit)
        {
            result[startIdx + bit] = static_cast<float>((byteValue & (uint8_t{1} << bit)) >> bit);
        }
    }
    return result;
}

std::array<uint8_t, 32> FeatureBoosterOnnxRunner::packbitsLittleEndian(const std::array<float, 256> &inData)
{
    std::array<uint8_t, 32> result;
    for (size_t i = 0; i < 32; ++i)
    {
        result[i] = 0;
        size_t startIdx = i * 8;
        for (auto bit = 0; bit < 8; ++bit)
        {
            uint8_t value = static_cast<uint8_t>(inData[startIdx + bit] >= 0.0 ? 1 : 0);
            result[i] |= (value << bit);
        }
    }

    return result;
}

FeatureBoosterOnnxRunner::FeatureBoosterOnnxRunner(unsigned int numThreads) : mnumThreads(numThreads)
{
}

FeatureBoosterOnnxRunner::~FeatureBoosterOnnxRunner()
{
}
