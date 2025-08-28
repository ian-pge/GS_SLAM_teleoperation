cmake_minimum_required(VERSION 3.5)
project(orb_slam3_ros2_wrapper)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(OpenGL_GL_PREFERENCE GLVND)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)

add_executable(
  rgbd
  src/rgbd.cpp
  src/rgbd-slam-node.cpp
  src/ros_utils.cpp
)

target_include_directories(
  rgbd
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CARTGS_ROOT_DIR}/viewer
  ${CARTGS_ROOT_DIR}
  ${CARTGS_ROOT_DIR}/include
  ${CARTGS_ROOT_DIR}/third_party
  ${ORB_SLAM3_ROOT_DIR}
  ${ORB_SLAM3_ROOT_DIR}/include
  ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
  ${ORB_SLAM3_ROOT_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(rgbd Eigen3 rclcpp sensor_msgs cv_bridge message_filters geometry_msgs nav_msgs tf2 tf2_ros)

target_link_libraries(
  rgbd
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CARTGS_ROOT_DIR}/lib/libgaussian_mapper.so
  ${CARTGS_ROOT_DIR}/lib/libgaussian_viewer.so
  ${ORB_SLAM3_ROOT_DIR}/lib/libORB_SLAM3.so
)

set_target_properties(rgbd PROPERTIES
    INSTALL_RPATH "${CARTGS_ROOT_DIR}/lib;${ORB_SLAM3_ROOT_DIR}/lib"
)

install(TARGETS rgbd DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
