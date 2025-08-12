using UnityEngine;

public class CameraMovement : MonoBehaviour
{
    public float moveSpeed = 10f; // Speed of camera movement
    public float rotationSpeed = 600f; // Speed of camera rotation

    private Camera cam;

    void Start()
    {
        cam = GetComponent<Camera>(); // Get the camera component
    }

    void Update()
    {
        HandleMovement();
        HandleRotation();
    }

    // Function to handle camera movement
    void HandleMovement()
    {
        // Get input for moving forward/backward (W/S) and left/right (A/D)
        float moveForward = Input.GetAxis("Vertical") * moveSpeed * Time.deltaTime; // W and S (or Arrow Up and Down)
        float moveRight = Input.GetAxis("Horizontal") * moveSpeed * Time.deltaTime; // A and D (or Arrow Left and Right)

        // Move the camera forward/backward and left/right relative to its current orientation
        cam.transform.Translate(new Vector3(moveRight, 0, moveForward));
    }

    // Function to handle camera rotation
    void HandleRotation()
    {
        // Rotate the camera around the Y axis (left/right) using mouse X input
        float rotateHorizontal = Input.GetAxis("Mouse X") * rotationSpeed * Time.deltaTime;

        // Rotate the camera around the X axis (up/down) using mouse Y input (inverted for standard FPS controls)
        float rotateVertical = -Input.GetAxis("Mouse Y") * rotationSpeed * Time.deltaTime;

        // Apply the rotation to the camera's transform
        cam.transform.Rotate(rotateVertical, rotateHorizontal, 0);

        // Prevent rolling (i.e., rotation around the Z axis) by resetting the local Z rotation to 0
        Vector3 currentRotation = cam.transform.localEulerAngles;
        cam.transform.localEulerAngles = new Vector3(currentRotation.x, currentRotation.y, 0);
    }
}
