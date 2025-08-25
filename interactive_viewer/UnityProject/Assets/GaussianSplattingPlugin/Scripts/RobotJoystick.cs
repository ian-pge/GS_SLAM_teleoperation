using UnityEngine;

public class GamepadController : MonoBehaviour
{
    [Header("Camera / Pivot References")]
    public Transform cameraTransform;  // Assign your main camera in the Inspector
    public Transform pivotTransform;   // Assign the "Pivot ghost" object in the Inspector

    [Header("Camera Settings")]
    public float cameraRotationSpeed = 100f; // Speed of camera rotation
    public float cameraDistance = 5f;        // Initial distance from the pivot to the camera
    public float zoomSpeed = 5f;             // Speed of zooming in and out
    public float minCameraDistance = 2f;     // Minimum camera distance from the pivot
    public float maxCameraDistance = 10f;    // Maximum camera distance from the pivot

    [Header("Movement Settings")]
    public float moveSpeed = 5f;             // Speed for pivot movement with right stick
    public float verticalMoveSpeed = 5f;     // Speed for vertical movement (via triggers)
    public float rotationSpeed = 100f;       // Speed for rotation around the pivot object's axis

    // Internal rotation tracking
    private float currentXRotation = 0f;     // Tracks the vertical rotation of the camera
    private float currentYRotation = 0f;     // Tracks the horizontal rotation of the camera

    void Update()
    {
        HandleCameraRotation();
        HandleCameraZoom();
        HandlePivotMovement();
        HandleVerticalMovement();
        HandleObjectRotation();
    }

    /// <summary>
    /// Rotates the camera around the pivot based on left stick input.
    /// </summary>
    void HandleCameraRotation()
    {
        // Left stick input
        float leftStickHorizontal = Input.GetAxis("Horizontal"); // Typically "Horizontal"
        float leftStickVertical   = Input.GetAxis("Vertical");   // Typically "Vertical"

        // Update camera rotation angles
        currentXRotation += leftStickVertical * cameraRotationSpeed * Time.deltaTime;
        currentYRotation -= leftStickHorizontal * cameraRotationSpeed * Time.deltaTime;

        // Clamp the vertical rotation to prevent flipping
        currentXRotation = Mathf.Clamp(currentXRotation, -85f, 85f);

        // Convert spherical to Cartesian coordinates
        float x = cameraDistance * Mathf.Cos(Mathf.Deg2Rad * currentXRotation)
                                * Mathf.Sin(Mathf.Deg2Rad * currentYRotation);
        float y = cameraDistance * Mathf.Sin(Mathf.Deg2Rad * currentXRotation);
        float z = cameraDistance * Mathf.Cos(Mathf.Deg2Rad * currentXRotation)
                                * Mathf.Cos(Mathf.Deg2Rad * currentYRotation);

        // Update camera position
        cameraTransform.position = pivotTransform.position + new Vector3(x, y, z);

        // Make the camera look at the pivot
        cameraTransform.LookAt(pivotTransform);
    }

    /// <summary>
    /// Zooms the camera in/out using A or B buttons.
    /// </summary>
    void HandleCameraZoom()
    {
        bool aButtonPressed = Input.GetButton("A"); // Typically "A" mapped in Input Manager
        bool bButtonPressed = Input.GetButton("B"); // Typically "B" mapped in Input Manager

        // Zoom in/out
        if (aButtonPressed)
        {
            cameraDistance -= zoomSpeed * Time.deltaTime;
        }
        if (bButtonPressed)
        {
            cameraDistance += zoomSpeed * Time.deltaTime;
        }

        // Clamp camera distance
        cameraDistance = Mathf.Clamp(cameraDistance, minCameraDistance, maxCameraDistance);
    }

    /// <summary>
    /// Moves the pivot (the "ghost" object) horizontally in the scene using the right stick.
    /// </summary>
    void HandlePivotMovement()
    {
        // Right stick input
        float rightStickHorizontal = Input.GetAxis("RightStickHorizontal"); 
        float rightStickVertical   = Input.GetAxis("RightStickVertical");

        // Movement in the XZ plane
        Vector3 movement = new Vector3(rightStickHorizontal, 0f, rightStickVertical)
                           * moveSpeed * Time.deltaTime;

        // Translate the pivot in world space
        pivotTransform.Translate(movement, Space.Self);
    }

    /// <summary>
    /// Uses separate triggers for vertical movement: 
    /// LeftTrigger (Axis 3) to move down, RightTrigger (Axis 4 or similar) to move up.
    /// </summary>
    void HandleVerticalMovement()
    {
        // IMPORTANT: Make sure these match your Input Manager axis names
        float leftTrigger  = Input.GetAxis("LeftTrigger");   // Mapped to Axis 3
        float rightTrigger = Input.GetAxis("RightTrigger");  // Mapped to Axis 4 (or chosen axis)

        // Right trigger is positive, left trigger is negative
        float verticalMovement = (rightTrigger - leftTrigger) * verticalMoveSpeed * Time.deltaTime;

        // Apply vertical movement
        pivotTransform.Translate(Vector3.up * verticalMovement, Space.World);
    }

    /// <summary>
    /// Rotates the pivot around its Z-axis using LB (left bumper) and RB (right bumper).
    /// </summary>
    void HandleObjectRotation()
    {
        bool leftBumper  = Input.GetButton("LeftBumper");  // Typically "LeftBumper"
        bool rightBumper = Input.GetButton("RightBumper"); // Typically "RightBumper"

        float rotation = 0f;
        if (leftBumper)
            rotation -= rotationSpeed * Time.deltaTime;
        if (rightBumper)
            rotation += rotationSpeed * Time.deltaTime;

        // Rotate around the Z-axis in World space
        // If you prefer local space, change Space.World to Space.Self.
        pivotTransform.Rotate(Vector3.up, rotation, Space.World);
    }
}
