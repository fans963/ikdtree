#pragma once

#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"
#include <mutex>

namespace config {
static std::unique_ptr<YAML::Node> root_ = std::make_unique<YAML::Node>();
static std::mutex mutex_;

static inline void load(const std::string& configFile) {
    std::lock_guard<std::mutex> locker(mutex_);
    *root_ = YAML::LoadFile(configFile);
    spdlog::info("Config file loaded from {}", configFile);
}

template <typename T, size_t N>
static inline auto value(const char* (&&keys)[N], const T&& default_value) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto node = YAML::Clone(*root_);
    try {
        for (const auto& key : keys)
            node = node[key];
        return node.as<T>();
    } catch (const YAML::Exception& e) {
        std::string log = "Failed to get value from key: ";
        for (const auto& key : keys) {
            log += "/";
            log += key;
        }
        spdlog::info(log);
        spdlog::info("Using default value: {}", default_value);
        return default_value;
    }
}

template <typename T, size_t N>
static inline void value(T&& t, const char* (&&keys)[N], const T&& default_value) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto node = YAML::Clone(*root_);
    try {
        for (const auto& key : keys)
            node = node[key];
        t = node.as<T>();
    } catch (const YAML::Exception& e) {
        std::string log = "Failed to get value from key: ";
        for (const auto& key : keys) {
            log += "/";
            log += key;
        }
        spdlog::info(log);
        t = default_value;
    }
}
} // namespace config