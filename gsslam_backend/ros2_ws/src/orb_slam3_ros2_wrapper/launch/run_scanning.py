import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # 1) ORB-SLAM3 RGB-D Node
    orb_slam = Node(
        package='orb_slam3_ros2_wrapper',
        executable='rgbd',
        arguments=[
            'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
            'cfg/ORB_SLAM3/RGB-D/RealCamera/zed_rgbd.yaml',
            'cfg/gaussian_mapper/RGB-D/RealCamera/zed_rgbd.yaml',
            'results'
        ],
        output='screen'
    )

    # 2) ROS TCP endpoint Node
    ros_tcp_endpoint_node = Node(
        package='ros_tcp_endpoint',
        executable='default_server_endpoint',
        output='screen',
        parameters=[{'ROS_IP': '0.0.0.0'}]
    )

    # 3) ZED wrapper launch file (delayed by 5 seconds)
    zed_camera_launch_path = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'launch',
        'zed_camera.launch.py'
    )

    zed_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_camera_launch_path),
        launch_arguments={
            'camera_model': 'zed2i',
            'svo_path': '/workspace/gsslam_backend/example.svo'
            # 'stream_address': "192.168.100.83"
        }.items()
    )

    delayed_zed_camera_launch = TimerAction(
        period=5.0,  # 5-second delay
        actions=[zed_camera_launch]
    )

    return LaunchDescription([
        orb_slam,
        ros_tcp_endpoint_node,
        delayed_zed_camera_launch
    ])
