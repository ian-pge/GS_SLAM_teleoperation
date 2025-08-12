#include <ros_utils.hpp>

using ImageMsg = sensor_msgs::msg::Image;
using PcdMsg = sensor_msgs::msg::PointCloud2;
using PoseMsg = geometry_msgs::msg::PoseStamped;
using OdomMsg = nav_msgs::msg::Odometry;

void publish_camera_pose(
  const rclcpp::Publisher<PoseMsg>::SharedPtr& publisher, 
  const rclcpp::Time& stamp,
  Sophus::SE3f& Twc,  
  std::string world_frame_id)
{
  // Twc: pose matrix from world coordinate to camera coordinate
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.frame_id = world_frame_id;
  pose_msg.header.stamp = stamp;

  pose_msg.pose.position.x = Twc.translation().x();
  pose_msg.pose.position.y = Twc.translation().y();
  pose_msg.pose.position.z = Twc.translation().z();

  pose_msg.pose.orientation.w = Twc.unit_quaternion().coeffs().w();
  pose_msg.pose.orientation.x = Twc.unit_quaternion().coeffs().x();
  pose_msg.pose.orientation.y = Twc.unit_quaternion().coeffs().y();
  pose_msg.pose.orientation.z = Twc.unit_quaternion().coeffs().z();

  publisher->publish(pose_msg);
}

void publish_body_odometry(
  const rclcpp::Publisher<OdomMsg>::SharedPtr& publisher,
  const rclcpp::Time& stamp, 
  Sophus::SE3f Twb, 
  Eigen::Vector3f Vwb, 
  Eigen::Vector3f Wwb,
  std::string world_frame_id,
  std::string odom_frame_id)
{
  // Twb: pose matrix from world coordinate to body coordinate
  // Vwb: body velocity in world coordinate
  // Wwb: body angular velocity in world coordinate
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.frame_id = world_frame_id;
  odom_msg.child_frame_id = odom_frame_id;
  odom_msg.header.stamp = stamp;

  // Coordinate Transform: OpenCV coordinate to ROS FLU coordinate
  Eigen::Matrix<float, 3, 3> cv_to_ros_rot; 
  Eigen::Matrix<float, 3, 1> cv_to_ros_trans; 
  cv_to_ros_rot << 0, -1, 0,
                  0, 0, -1,
                  1, 0, 0;
  cv_to_ros_trans << 0, 0, 0;
  Sophus::SE3f cv_to_ros(cv_to_ros_rot, cv_to_ros_trans);

  Twb = cv_to_ros * Twb; 
  odom_msg.pose.pose.position.x = Twb.translation().x();
  odom_msg.pose.pose.position.y = Twb.translation().y();
  odom_msg.pose.pose.position.z = Twb.translation().z();

  odom_msg.pose.pose.orientation.w = Twb.unit_quaternion().coeffs().w();
  odom_msg.pose.pose.orientation.x = Twb.unit_quaternion().coeffs().x();
  odom_msg.pose.pose.orientation.y = Twb.unit_quaternion().coeffs().y();
  odom_msg.pose.pose.orientation.z = Twb.unit_quaternion().coeffs().z();

  Vwb = cv_to_ros_rot * Vwb; 
  Wwb = cv_to_ros_rot * Wwb; 

  odom_msg.twist.twist.linear.x = Vwb.x();
  odom_msg.twist.twist.linear.y = Vwb.y();
  odom_msg.twist.twist.linear.z = Vwb.z();

  odom_msg.twist.twist.angular.x = Wwb.x();
  odom_msg.twist.twist.angular.y = Wwb.y();
  odom_msg.twist.twist.angular.z = Wwb.z();

  publisher->publish(odom_msg);
}

void publish_tracking_img(
  const rclcpp::Publisher<ImageMsg>::SharedPtr& publisher,
  const rclcpp::Time& stamp,
  cv::Mat image, 
  std::string frame_id)
{
  std_msgs::msg::Header header;

  header.stamp = stamp;
  header.frame_id = frame_id;

  sensor_msgs::msg::Image::SharedPtr image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

  publisher->publish(*image_msg.get());
}

void publish_camera_tf(
  const std::shared_ptr<tf2_ros::TransformBroadcaster>& tf_broadcaster,
  const rclcpp::Time& stamp,
  Sophus::SE3f& Twc,
  std::string parent_frame_id,
  std::string child_frame_id)
{
  // T: pose matrix from parent frame to child frame
  geometry_msgs::msg::TransformStamped t;

  t.header.stamp = stamp;
  t.header.frame_id = parent_frame_id;
  t.child_frame_id = child_frame_id;

  t.transform.translation.x = Twc.translation().x();
  t.transform.translation.y = Twc.translation().y();
  t.transform.translation.z = Twc.translation().z();

  t.transform.rotation.w = Twc.unit_quaternion().coeffs().w();
  t.transform.rotation.x = Twc.unit_quaternion().coeffs().x();
  t.transform.rotation.y = Twc.unit_quaternion().coeffs().y();
  t.transform.rotation.z = Twc.unit_quaternion().coeffs().z();

  tf_broadcaster->sendTransform(t);
}


void publish_world_to_odom_tf(
  const std::shared_ptr<tf2_ros::TransformBroadcaster>& tf_broadcaster,
  const rclcpp::Time& stamp,
  Sophus::SE3f& Two, 
  std::string parent_frame_id,
  std::string child_frame_id)
{
  // T: pose matrix from parent frame to child frame
  geometry_msgs::msg::TransformStamped t;

  t.header.stamp = stamp;
  t.header.frame_id = parent_frame_id;
  t.child_frame_id = child_frame_id;

  t.transform.translation.x = Two.translation().x();
  t.transform.translation.y = Two.translation().y();
  t.transform.translation.z = Two.translation().z();

  t.transform.rotation.w = Two.unit_quaternion().coeffs().w();
  t.transform.rotation.x = Two.unit_quaternion().coeffs().x();
  t.transform.rotation.y = Two.unit_quaternion().coeffs().y();
  t.transform.rotation.z = Two.unit_quaternion().coeffs().z();

  tf_broadcaster->sendTransform(t);
}


Eigen::Affine3f transform_to_eigen(
  const geometry_msgs::msg::TransformStamped& transform_stamped){
    Eigen::Vector3f translation(
      transform_stamped.transform.translation.x, 
      transform_stamped.transform.translation.y, 
      transform_stamped.transform.translation.z
    ); 

    Eigen::Quaternionf quat(
      transform_stamped.transform.rotation.w, 
      transform_stamped.transform.rotation.x, 
      transform_stamped.transform.rotation.y, 
      transform_stamped.transform.rotation.z
    ); 
    
    Eigen::Translation<float, 3> trans(translation); 
    Eigen::Matrix3f rot = quat.toRotationMatrix(); 
    Eigen::Affine3f affine = trans * rot; 

    return affine; 
  }


Sophus::SE3f transform_to_SE3(
  const geometry_msgs::msg::TransformStamped& transform_stamped){

    Eigen::Vector3f translation(
      transform_stamped.transform.translation.x, 
      transform_stamped.transform.translation.y, 
      transform_stamped.transform.translation.z
    ); 

    Eigen::Quaternionf quat(
      transform_stamped.transform.rotation.w, 
      transform_stamped.transform.rotation.x, 
      transform_stamped.transform.rotation.y, 
      transform_stamped.transform.rotation.z
    ); 

    Eigen::Matrix3f rotation = quat.toRotationMatrix(); 
    
    Sophus::SE3f se3(rotation, translation);

    return se3;
  }