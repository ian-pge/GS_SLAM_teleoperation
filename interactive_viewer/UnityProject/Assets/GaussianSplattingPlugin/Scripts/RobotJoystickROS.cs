using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using PoseStampedMsg = RosMessageTypes.Geometry.PoseStampedMsg;

public class GoalMover : MonoBehaviour
{
    [SerializeField]
    private string goalTopic = "goal";

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<PoseStampedMsg>(goalTopic, OnGoalReceived);
    }

    private void OnGoalReceived(PoseStampedMsg msg)
    {
        transform.position = msg.pose.position.From<FLU>();
        transform.rotation = msg.pose.orientation.From<FLU>();
    }
}
