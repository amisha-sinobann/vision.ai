# Vision OS - Local ML Server

This project includes a Python server for local object detection using YOLOv8, specifically designed to run alongside the React dashboard.

## Prerequisites

1.  **Python 3.8+** installed.
2.  **ESP32-CAM** streaming MJPEG at `http://192.168.1.6:81/stream` (or update the IP in the script).

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the server:
    ```bash
    python vision_os_server_local_ml.py
    ```
    *On the first run, it will automatically download the `yolov8n.pt` model.*

## Features

*   **Object Detection**: Uses YOLOv8 to detect 80+ classes of objects.
*   **Color Detection**: Identifies dominant colors of detected objects.
*   **Direction Analysis**: Determines if objects are to the left, right, or center.
*   **Currency Detection**: (Stubbed) Logic for detecting currency.
*   **API**: Exposes a Flask API at `http://localhost:5000` for the React dashboard.

## Dashboard Integration

The React dashboard is configured to connect to `http://localhost:5000`.
*   **Video Feed**: Proxied via `http://localhost:5000/frame` to avoid CORS issues with ESP32.
*   **Data Stream**: Uses Server-Sent Events (SSE) at `http://localhost:5000/events` for real-time updates.
