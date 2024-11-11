import cv2
from ultralytics import YOLO

# Ensure CUDA is available
import torch
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your GPU installation.")
    exit()

# Load your ONNX YOLOv8 model
model = YOLO("/path to directory/best.onnx")  # Replace with your ONNX model path

# Open a video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the video capture is opened
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Loop for processing the video feed
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from webcam.")
            break

        # Perform detection on the frame with GPU
        results = model.predict(source=frame, conf=0.5, device=0)  # Use device=0 for GPU

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Detection (GPU)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nExiting...")

# Release resources
cap.release()
cv2.destroyAllWindows()
