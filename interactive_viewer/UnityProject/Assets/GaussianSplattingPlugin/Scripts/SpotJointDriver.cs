using System.Collections.Generic;
using UnityEngine;
#if ROS2        // <- optional: comment out if you do not use ROS
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
#endif

public class SpotTransformDriver : MonoBehaviour
{
    // ------------------------------------------------------------------
    //  Joint specification  (name + hinge axis)  ••• NO INSPECTOR WORK •••
    // ------------------------------------------------------------------
    struct Spec { public string name; public Vector3 axis;
                  public Spec(string n, Vector3 a){name=n; axis=a;} }

    static readonly Spec[] Specs = {
        new("fl.hx",  Vector3.right), new("fl.hy",  Vector3.up),
        new("fl.kn",  Vector3.up),
        new("fr.hx",  Vector3.right), new("fr.hy",  Vector3.up),
        new("fr.kn",  Vector3.up),
        new("hl.hx",  Vector3.right), new("hl.hy",  Vector3.up),
        new("hl.kn",  Vector3.up),
        new("hr.hx",  Vector3.right), new("hr.hy",  Vector3.up),
        new("hr.kn",  Vector3.up),
        new("arm0.shoulder_yaw",   Vector3.forward),
        new("arm0.shoulder_pitch", Vector3.up),
        new("arm0.elbow_pitch",    Vector3.up),
        new("arm0.elbow_roll",     Vector3.right),
        new("arm0.wrist_pitch",    Vector3.up)
        // (fixed joints like wrist_roll / payload are ignored)
    };

    //  name → (Transform, axis) fast lookup
    Dictionary<string,(Transform tf,Vector3 ax)> map = new ();

    // ---------- INITIALISE ------------------------------------------------
    void Awake()
    {
        foreach (var s in Specs)
        {
            Transform t = transform.Find(s.name);   // deep-find by name
            if (t == null)                          // URDF importer keeps names
                t = SearchRecursive(transform, s.name);
            if (t != null) map[s.name] = (t, s.axis);
            else           Debug.LogWarning($"Joint '{s.name}' not found");
        }
    }

    // ---------- PUBLIC API -------------------------------------------------
    /// <summary>Set a joint’s absolute angle (radians).</summary>
    public void SetJoint(string name, float rad)
    {
        if (!map.TryGetValue(name, out var v)) return;
        v.tf.localRotation =
            Quaternion.AngleAxis(rad * Mathf.Rad2Deg, v.ax);   // Unity API :contentReference[oaicite:5]{index=5}
    }

    // ---------- OPTIONAL: ROS-2 subscriber --------------------------------
#if ROS2
    void Start()
    {
        ROSConnection.GetOrCreateInstance()
                     .Subscribe<JointStateMsg>("/spot/joint_states", OnJointState);
    }
    void OnJointState(JointStateMsg msg)
    {
        for (int i = 0; i < msg.name.Length; i++)
            SetJoint(msg.name[i], (float)msg.position[i]);     // radians from ROS
    }
#endif

    // ---------- OPTIONAL: quick keyboard test ----------------------------
    void Update()
    {
        float d = Input.GetAxis("Horizontal") * 1.0f * Time.deltaTime;
        if (Mathf.Abs(d) > 1e-4f) SetJoint("fl.hx", _Get("fl.hx") + d);
    }
    float _Get(string n) => map.TryGetValue(n, out var v)
                                ? Quaternion.Angle(v.tf.localRotation,
                                                    Quaternion.identity)*Mathf.Deg2Rad
                                : 0f;

    // ---------- helper ----------------------------------------------------
    static Transform SearchRecursive(Transform root,string n)
    {
        foreach (Transform c in root)
            if (c.name == n) return c;
            else { var r = SearchRecursive(c,n); if (r) return r; }
        return null;
    }
}
