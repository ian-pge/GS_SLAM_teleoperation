using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using PoseStampedMsg = RosMessageTypes.Geometry.PoseStampedMsg;

public class PoseSubscriber : MonoBehaviour
{
    [SerializeField]
    private string topicName = "camera_pose";

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<PoseStampedMsg>(topicName, PoseCallback);
    }

    private void PoseCallback(PoseStampedMsg poseMsg)
    {
        transform.position = poseMsg.pose.position.From<FLU>();
        transform.rotation = poseMsg.pose.orientation.From<FLU>();
    }
}
