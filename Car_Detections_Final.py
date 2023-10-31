import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

#Tracker
tracker = sv.ByteTrack()

#Open the video file
video_path = 'cars.mp4' 
#video_path = 0 
cap = cv2.VideoCapture(video_path)
# frames_generator = sv.get_video_frames_generator(video_path)
# fps_monitor = sv.FPSMonitor()

#Options
selected_classes=[0,1,2,3,5,6,7]
confidence_threshold= 0.4
iou_threshold= 0.7

#box annotator
box_annotator = sv.BoundingBoxAnnotator()

#Label annotator
label_annotator = sv.LabelAnnotator(text_scale=0.25, text_padding=5, text_position=sv.Position.TOP_CENTER)

#Loop through the video frames
while cap.isOpened():
    #Read a frame from the video
    success, frame = cap.read()

    if success:
        #Run YOLOv8 tracking on the frame, persisting tracks between frames
        
        results = model.track(frame, conf=confidence_threshold, iou=iou_threshold, agnostic_nms=True, persist=True,)[0]
        detections = sv.Detections.from_ultralytics(results)
        #Adding type of tracker
        detections = tracker.update_with_detections(detections)
        #Class detections
        detections = detections[np.isin(detections.class_id, selected_classes)]
        
        #Labels
        labels = [
            f"{model.model.names[class_id]} {tracker_id}  {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]
        
        
        #Visualize the results on a box
        annotated_frame = box_annotator.annotate(
            scene=frame, 
            detections=detections
        )
        #Visualize the labels
        annotated_labeled_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections,
            labels=labels
        )
        
        #fps_monitor
        # fps_monitor.tick()
        # fps = fps_monitor()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        #Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        #Break the loop if the end of the video is reached
        break

#Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
