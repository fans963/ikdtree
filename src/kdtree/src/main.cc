#include "ikdtree/ikdtree.hh"
#include "config/config.hh"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <csignal>
#include <filesystem>
#include "eigen3/Eigen/Eigen"

int main() {
    static std::atomic<bool> running(true);
    std::signal(SIGEV_SIGNAL, [](int sig) { running = false; });

    auto configPath = std::filesystem::canonical("/proc/self/exe")
                          .parent_path()
                          .parent_path()
                          .parent_path()
                          .string();
    configPath += "/include/config/config.yaml";
    config::load(configPath);

    // 测试数据：三维空间中的点
    std::vector<Eigen::Vector3f> points = {{2, 3, 1}, {5, 4, 2}, {9, 6, 3},
                            {4, 7, 5}, {8, 1, 6}, {7, 2, 4}};

    KdTree::KdTree<Eigen::Vector3f> wadwd(0, 0, 0);
    wadwd.range();

    return 0;
}