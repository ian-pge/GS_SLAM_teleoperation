#include "rgbd-slam-node.hpp"
#include <opencv2/core/core.hpp>
#include <filesystem> // Ensure C++17 or later is used

RgbdCartgsNode::RgbdCartgsNode(const std::string &vocab_file,
                               const std::string &orb_settings_file,
                               const std::string &gaussian_settings_file,
                               const std::string &output_dir,
                               bool use_viewer)
: Node("cartgs_orb_slam_ros2"),
  m_output_dir(output_dir),
  m_use_viewer(use_viewer)
{
    if (m_output_dir.back() != '/')
        m_output_dir += "/";

    // Determine device type for GaussianMapper
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        RCLCPP_INFO(this->get_logger(), "CUDA available! Using GPU.");
        device_type = torch::kCUDA;
    } else {
        RCLCPP_INFO(this->get_logger(), "Using CPU.");
        device_type = torch::kCPU;
    }

    // Create ORB-SLAM3 system
    m_SLAM = std::make_shared<ORB_SLAM3::System>(vocab_file, orb_settings_file, ORB_SLAM3::System::RGBD);

    float imageScale = m_SLAM->GetImageScale();

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(gaussian_settings_file);
    m_pGausMapper = std::make_shared<GaussianMapper>(m_SLAM, gaussian_cfg_path, std::filesystem::path(m_output_dir), 0, device_type);
    m_training_thd = std::thread(&GaussianMapper::run, m_pGausMapper.get());

    // Optional Viewer
    if (m_use_viewer) {
        m_pViewer = std::make_shared<ImGuiViewer>(m_SLAM, m_pGausMapper);
        m_viewer_thd = std::thread(&ImGuiViewer::run, m_pViewer.get());
    }

    // Subscribe to RGB and Depth topics
    rgb_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "/zed/zed_node/left/image_rect_color");
    depth_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "/zed/zed_node/depth/depth_registered");

    pubPose_ = this->create_publisher<PoseMsg>("camera_pose", 1);
    pubTrackImage_ = this->create_publisher<ImageMsg>("tracking_image", 1);
    pubPcd_ = this->create_publisher<PcdMsg>("point_cloud", 1);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    syncApproximate = std::make_shared<message_filters::Synchronizer<approximate_sync_policy>>(approximate_sync_policy(10), *rgb_sub, *depth_sub);

    // Register the callback with ConstSharedPtr
    syncApproximate->registerCallback(
        std::bind(&RgbdCartgsNode::GrabRGBD, this, std::placeholders::_1, std::placeholders::_2)
    );

    RCLCPP_INFO(this->get_logger(), "cartgs ORB-SLAM node initialized and waiting for images...");
}

RgbdCartgsNode::~RgbdCartgsNode()
{
    // Stop all threads
    m_SLAM->Shutdown();
    m_training_thd.join();
    if (m_use_viewer && m_viewer_thd.joinable()) {
        m_viewer_thd.join();
    }

    // Save camera trajectory
    m_SLAM->SaveTrajectoryTUM(m_output_dir + "CameraTrajectory_TUM.txt");
    m_SLAM->SaveKeyFrameTrajectoryTUM(m_output_dir + "KeyFrameTrajectory_TUM.txt");
    m_SLAM->SaveTrajectoryEuRoC(m_output_dir + "CameraTrajectory_EuRoC.txt");
    m_SLAM->SaveKeyFrameTrajectoryEuRoC(m_output_dir + "KeyFrameTrajectory_EuRoC.txt");
    m_SLAM->SaveTrajectoryKITTI(m_output_dir + "CameraTrajectory_KITTI.txt");

    RCLCPP_INFO(this->get_logger(), "Shut down cartgs ORB-SLAM node and saved trajectories.");
}

void RgbdCartgsNode::GrabRGBD(const ImageMsg::ConstSharedPtr &msgRGB,
                              const ImageMsg::ConstSharedPtr &msgD)
{
    // Convert ROS images to cv::Mat
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(),
                     "cv_bridge exception (RGB): %s", e.what());
        return;
    }

    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(),
                     "cv_bridge exception (Depth): %s", e.what());
        return;
    }

    // Convert BGRA → RGB if needed
    cv::Mat image_rgb;
    if (cv_ptrRGB->image.channels() == 4)
    {
        // If the image has 4 channels (BGRA), convert it to a 3-channel RGB
        cv::cvtColor(cv_ptrRGB->image, image_rgb, cv::COLOR_BGRA2RGB);
    }
    else if (cv_ptrRGB->image.channels() == 3)
    {
        // If already 3 channels (BGR), then convert BGR → RGB
        cv::cvtColor(cv_ptrRGB->image, image_rgb, cv::COLOR_BGR2RGB);
    }
    else
    {
        // Otherwise handle unexpected channel count
        RCLCPP_ERROR(this->get_logger(),
                     "Unexpected number of channels in color image: %d",
                     cv_ptrRGB->image.channels());
        return;
    }

    // Use the timestamp from the incoming RGB image
    double tframe = Utility::StampToSec(msgRGB->header.stamp);

    // Run ORB-SLAM3 + GaussianMapper tracking
    // Pass 'image_rgb' (now 3-channel) instead of cv_ptrRGB->image
    Sophus::SE3f Tcw;
    Tcw = m_SLAM->TrackRGBD(image_rgb, cv_ptrD->image, tframe,
                      std::vector<ORB_SLAM3::IMU::Point>(), "");


    Sophus::SE3f Twc = Tcw.inverse(); // camera optical frame pose in opencv coordinate

    // Hardcoded frame names
    std::string world_frame = "map";
    std::string odom_frame = "odom";
    std::string camera_frame = "camera";
    std::string camera_optical_frame = "camera_optical_link";

    // define coordinate transforms ///
    // OpenCV to ROS FLU coordinate transforms
    Eigen::Matrix<float, 3, 3> cv_to_ros_rot;
    Eigen::Matrix<float, 3, 1> cv_to_ros_trans;
    cv_to_ros_rot << 0, 0, 1,
                    -1, 0, 0,
                     0,-1, 0;
    cv_to_ros_trans << 0, 0, 0;
    Sophus::SE3f cv_to_ros(cv_to_ros_rot, cv_to_ros_trans);
    // std::cout << cv_to_ros.matrix() << std::endl;

    // Coordinate Transform: OpenCV coordinate to ROS FLU coordinate
    Twc = cv_to_ros * Twc;
    Twc = Twc * cv_to_ros.inverse();

    // Convert degrees to radians
    float camera_rotation_deg = 0.0f; // degrees
    float camera_rotation = camera_rotation_deg * M_PI / 180.0f; // convert to radians

    Eigen::Matrix<float, 3, 3> cam_to_drone_rot;
    Eigen::Matrix<float, 3, 1> cam_to_drone_trans;

    cam_to_drone_rot << cos(camera_rotation), 0, sin(camera_rotation),
                        0,                 1,  0,
                        -sin(camera_rotation), 0,  cos(camera_rotation);
    cam_to_drone_trans << 0.0f, 0.0f, 0.0f;

    Sophus::SE3f cam_to_drone(cam_to_drone_rot, cam_to_drone_trans);

    Twc = cam_to_drone * Twc;
    Twc = Twc * cam_to_drone.inverse();

    // Option2: publish map to camera tf from SLAM
    publish_camera_tf(tf_broadcaster_, this->get_clock()->now(), Twc, world_frame, camera_frame);
    publish_camera_pose(pubPose_, this->get_clock()->now(), Twc, world_frame);
    // publish_tracking_img(pubTrackImage_, this->get_clock()->now(), m_SLAM->GetCurrentFrame(), world_frame); NOT WORKING RIGHT NOW
}
