using UnityEngine;

public class CubeMovementWithRotation : MonoBehaviour
{
    public float moveSpeed = 5f;       // Speed of robot movement
    public float rotationSpeed = 100f; // Speed of robot rotation

    void Update()
    {
        HandleMovement();
        HandleRotation();
    }

    void HandleMovement()
    {
        Vector3 moveDirection = Vector3.zero;

        // Move forward along local Z-axis when 'I' is pressed
        if (Input.GetKey(KeyCode.I))
        {
            moveDirection += Vector3.forward;
        }
        // Move backward along local Z-axis when 'K' is pressed
        if (Input.GetKey(KeyCode.K))
        {
            moveDirection += Vector3.back;
        }
        // Move up along local Y-axis when 'U' is pressed
        if (Input.GetKey(KeyCode.U))
        {
            moveDirection += Vector3.up;
        }
        // Move down along local Y-axis when 'O' is pressed
        if (Input.GetKey(KeyCode.O))
        {
            moveDirection += Vector3.down;
        }

        // Normalize to prevent faster diagonal movement
        if (moveDirection != Vector3.zero)
        {
            moveDirection = moveDirection.normalized * moveSpeed * Time.deltaTime;

            // Apply movement to the robot along its local axes
            transform.Translate(moveDirection, Space.Self);
        }
    }

    void HandleRotation()
    {
        float rotationY = 0f;

        // Rotate around local Y-axis (Yaw) with 'J' and 'L'
        if (Input.GetKey(KeyCode.J))
        {
            rotationY -= rotationSpeed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.L))
        {
            rotationY += rotationSpeed * Time.deltaTime;
        }

        // Apply rotation around the robot's local Y-axis
        if (rotationY != 0f)
        {
            transform.Rotate(0f, rotationY, 0f);
        }
    }
}
