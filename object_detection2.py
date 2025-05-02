import cv2
from ultralytics import YOLO

def initialize_model(model_path='yolov8s.pt'):
    """
    Load a YOLOv8 object detection model.
    """
    return YOLO(model_path)

def annotate_frame(frame, predictions, label_list):
    """
    Draw bounding boxes and class labels on the frame.
    """
    for item in predictions.boxes:
        x_start, y_start, x_end, y_end = map(int, item.xyxy[0])
        category_id = int(item.cls[0])
        score = float(item.conf[0])
        label = label_list[category_id]
        
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{label} {score:.2f}", (x_start, y_start - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
    return frame

def start_video_detection():
    """
    Launch webcam-based object detection.
    """
    detector = initialize_model()
    label_names = detector.names

    stream = cv2.VideoCapture(0)
    if not stream.isOpened():
        print("Camera could not be accessed.")
        return

    print("Press 'q' to quit.")
    while True:
        active, frame = stream.read()
        if not active:
            break

        outcomes = detector(frame)[0]
        frame_with_labels = annotate_frame(frame, outcomes, label_names)

        cv2.imshow("Live Object Detection", frame_with_labels)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video_detection()
