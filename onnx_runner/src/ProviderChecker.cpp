/**
 * @file ProviderChecker.cpp
 * @author letterso
 * @brief 获取支持的providers
 * @version 0.1
 * @date 2023-12-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main()
{
    std::vector<std::string> providers = Ort::GetAvailableProviders();

    for (const auto &provider: providers) {
        std::cout << provider << std::endl;
    }

    return 0;
}