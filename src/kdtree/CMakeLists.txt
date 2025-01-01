cmake_minimum_required(VERSION 3.8)
project(ikdtree)

set(CMAKE_C_COMPILER clang-20)
set(CMAKE_CXX_COMPILER clang++-20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c26")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++26")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -Os -Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(${PROJECT_NAME}_exe src/main.cc)

ament_target_dependencies(${PROJECT_NAME}_exe
  rclcpp
  pcl_ros
  pcl_conversions
  PCL 
  Eigen3
)

target_link_libraries(${PROJECT_NAME}_exe
  spdlog::spdlog
  yaml-cpp
)

include_directories(${PROJECT_SOURCE_DIR}/include)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
