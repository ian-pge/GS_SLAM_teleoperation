using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using PoseStampedMsg = RosMessageTypes.Geometry.PoseStampedMsg;

public class CameraGoalMover : MonoBehaviour
{
    [SerializeField]
    private string cameraGoalTopic = "camera_goal";

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<PoseStampedMsg>(cameraGoalTopic, OnCameraGoalReceived);
    }

    private void OnCameraGoalReceived(PoseStampedMsg msg)
    {
        transform.position = msg.pose.position.From<FLU>();
        transform.rotation = msg.pose.orientation.From<FLU>();
    }
}
