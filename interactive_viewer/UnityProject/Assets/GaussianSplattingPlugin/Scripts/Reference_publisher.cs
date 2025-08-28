using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class PosePublisher : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "/reference";
    public GameObject targetObject;
    public float publishMessageFrequency = 0.5f;
    private float timeElapsed;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PoseMsg>(topicName);
    }

    void Update()
    {
        timeElapsed += Time.deltaTime;
        if (timeElapsed > publishMessageFrequency)
        {
            Vector3 position = targetObject.transform.position;
            Quaternion rotation = targetObject.transform.rotation;

            PoseMsg poseMsg = new PoseMsg
            {
                position = new PointMsg(position.x, position.y, position.z),
                orientation = new QuaternionMsg(rotation.x, rotation.y, rotation.z, rotation.w)
            };

            ros.Publish(topicName, poseMsg);
            timeElapsed = 0;
        }
    }
}
