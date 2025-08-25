using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathDrawer : MonoBehaviour
{
    // Interval between samples (in seconds).
    public float sampleInterval = 0.05f;
    
    // Control the line width via the Inspector.
    public float lineWidth = 0.1f;
    
    // Set the line color to electric purple.
    // Electric purple RGB: approximately (0.75, 0.0, 1.0)
    public Color lineColor = new Color(0.75f, 0f, 1f, 1f);

    // List to store the object's positions.
    private List<Vector3> points = new List<Vector3>();
    
    // The LineRenderer component that will draw the path.
    private LineRenderer lineRenderer;

    void Start()
    {
        // Get or add a LineRenderer component.
        lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
        }
        
        // Configure the LineRenderer properties.
        lineRenderer.startWidth = lineWidth;
        lineRenderer.endWidth = lineWidth;
        // Assign a simple material; you can also assign one through the Inspector.
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        
        // Set the color of the line.
        lineRenderer.startColor = lineColor;
        lineRenderer.endColor = lineColor;
        lineRenderer.material.color = lineColor;
        
        lineRenderer.positionCount = 0;
        
        // Begin recording the object's path.
        StartCoroutine(RecordPath());
    }

    IEnumerator RecordPath()
    {
        while (true)
        {
            // Add the current position to the points list.
            points.Add(transform.position);
            
            // Update the LineRenderer with the new set of points.
            lineRenderer.positionCount = points.Count;
            lineRenderer.SetPositions(points.ToArray());
            
            // Wait for the specified interval before recording the next point.
            yield return new WaitForSeconds(sampleInterval);
        }
    }
}
