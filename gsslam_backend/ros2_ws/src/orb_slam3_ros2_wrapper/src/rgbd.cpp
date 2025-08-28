#include "rclcpp/rclcpp.hpp"
#include "rgbd-slam-node.hpp"

// colcon build --cmake-args -DCMAKE_CXX_FLAGS="-w" --symlink-install --packages-select orbslam3_ros2_wrapper
// Usage: ros2 run orb_slam3_ros2_wrapper rgbd third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt cfg/ORB_SLAM3/RGB-D/RealCamera/zed_rgbd.yaml cfg/gaussian_mapper/RGB-D/RealCamera/zed_rgbd.yaml results no_viewer

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::string vocab_file = argv[1];
    std::string orb_settings_file = argv[2];
    std::string gaussian_settings_file = argv[3];
    std::string output_dir = argv[4];

    // Check the last argument for "no_viewer"
    bool use_viewer = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "no_viewer") {
            use_viewer = false;
            break;
        }
    }

    auto node = std::make_shared<RgbdCartgsNode>(vocab_file, orb_settings_file, gaussian_settings_file, output_dir, use_viewer);
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}


// #colcon build vision_opencv
// #colcon build --packages-select cv_bridge