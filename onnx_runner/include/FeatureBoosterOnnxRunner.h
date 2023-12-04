/**
 * @file main.cpp
 * @author letterso
 * @brief
 * @version 0.1
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#ifndef FEATUREBOOSTER_ONNX_RUNNER_H
#define FEATUREBOOSTER_ONNX_RUNNER_H

#include <iostream>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <memory>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// #include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

class FeatureBoosterOnnxRunner
{
private:
    const unsigned int mnumThreads;

    // ONNX
    Ort::Env env;
    Ort::SessionOptions msessionOptions;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char *> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;
    std::vector<char *> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;

    std::vector<Ort::Value> minferOutputtensors;

    int env_device_ = 0; // 0 CPU, 1 GPU
private:
    int PreProcess(const cv::Size &imageSize, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &desc, std::vector<std::array<float, 4>> &kpts, std::vector<std::array<float, 256>> &desc_pre);
    int Inference(const std::vector<std::array<float, 4>> &kpts, const std::vector<std::array<float, 256>> &desc_pre);
    int PostProcess(cv::Mat &desc_boost);

    // 实现numpy的unpackbits和packbits
    std::array<float, 256> unpackbitsLittleEndian(const cv::Mat &inData);
    std::array<uint8_t, 32> packbitsLittleEndian(const std::array<float, 256> &inData);

public:
    FeatureBoosterOnnxRunner(unsigned int numThreads = 1);
    ~FeatureBoosterOnnxRunner();

    // 解决多线程时冲突导致的程序崩溃
    mutable std::mutex mtx_feature_booster;

    /**
     * @brief ONNX初始化
     *
     * @param onnx_file
     * @param env_device 0 CPU 1 GPU
     * @return int
     */
    int InitOrtEnv(const std::string &onnx_file, const unsigned int &env_device = 0);

    /**
     * @brief 推理
     *
     * @param keypoints
     * @param desc
     * @param descBoost
     * @return int
     */
    int InferenceDescriptor(const cv::Size &imageSize, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &desc, cv::Mat &descBoost);
};

#endif // FEATUREBOOSTER_ONNX_RUNNER_H