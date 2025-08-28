#ifndef __RGBD_CARTGS_NODE_HPP__
#define __RGBD_CARTGS_NODE_HPP__

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "utility.hpp"
#include "ros_utils.hpp"

// Include cartgs and ORB-SLAM3 related headers
#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "gaussian_mapper.h"
#include "imgui_viewer.h"

class RgbdCartgsNode : public rclcpp::Node
{
public:
    RgbdCartgsNode(const std::string &vocab_file, 
                   const std::string &orb_settings_file,
                   const std::string &gaussian_settings_file,
                   const std::string &output_dir,
                   bool use_viewer);

    ~RgbdCartgsNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    typedef message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg> approximate_sync_policy;

    // Use ConstSharedPtr in the callback signature
    void GrabRGBD(const ImageMsg::ConstSharedPtr &msgRGB, const ImageMsg::ConstSharedPtr &msgD);

    // ORB-SLAM3 system
    std::shared_ptr<ORB_SLAM3::System> m_SLAM;

    // Gaussian mapper and viewer
    std::shared_ptr<GaussianMapper> m_pGausMapper;
    std::shared_ptr<ImGuiViewer> m_pViewer;
    std::thread m_training_thd;
    std::thread m_viewer_thd;
    bool m_use_viewer;

    // Subscriptions and publishers
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImageConstPtr cv_ptrD;

    std::shared_ptr<message_filters::Subscriber<ImageMsg>> rgb_sub;
    std::shared_ptr<message_filters::Subscriber<ImageMsg>> depth_sub;
    std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy>> syncApproximate;

    rclcpp::Publisher<PoseMsg>::SharedPtr pubPose_;
    rclcpp::Publisher<PcdMsg>::SharedPtr pubPcd_;
    rclcpp::Publisher<ImageMsg>::SharedPtr pubTrackImage_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_; 
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_; 

    std::string m_output_dir;
};

#endif