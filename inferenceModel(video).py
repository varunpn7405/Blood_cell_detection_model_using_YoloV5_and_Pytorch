import torch
import cv2
import numpy as np
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

cls_clr={"RBC":(0,0,255),"WBC":(255,0,0),"Platelets":(0,255,0)}
# Load the YOLOv5 model
model_path = 'blood_cell_detection.pt'  # Replace with your model path
model = torch.hub.load('yolov5-master', 'custom', path=model_path, source='local',force_reload=True) 

# Open the video file
video_path = r"Trimmed_video\images.mp4"
cap = cv2.VideoCapture(video_path)
output_path="./predicted.mp4"

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for better speed
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret,frame=cap.read()

    if not ret:
        break
    
    # Convert the image to RGB (YOLOv5 expects RGB images)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Print results
    results.print()  # Print results to the console

    # Extract bounding boxes and labels
    bbox_tensor = results.xyxy[0].cpu()  # Move the tensor to CPU
    bbox_array = bbox_tensor.numpy()  # Convert to NumPy array

    # Non-Max Suppression parameters (if you want to customize)
    iou_threshold = 0.5  # Intersection over Union threshold
    confidence_threshold = 0.25  # Confidence score threshold

    # Apply NMS
    filtered_boxes = []
    for bbox in bbox_array:
        if bbox[4] >= confidence_threshold:
            filtered_boxes.append(bbox)

    filtered_boxes = np.array(filtered_boxes)

    # Convert to the format required by cv2.dnn.NMSBoxes
    boxes = [[int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])] for box in filtered_boxes]
    scores = [box[4] for box in filtered_boxes]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=confidence_threshold,
        nms_threshold=iou_threshold
    )

    # Draw bounding boxes and labels
    for i in indices:

        x1, y1, w, h = boxes[i]
        x2, y2 = x1 + w, y1 + h
        conf, cls = filtered_boxes[i][4], filtered_boxes[i][5]
        label = model.names[int(cls)]
        conf = f'{conf:.2f}'
        classTitle=f"{label} {conf}"
        clr=cls_clr[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 1)
        text_size = cv2.getTextSize(classTitle, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1]), (x1 + text_size[0], y1), clr, -1)
        cv2.putText(frame, classTitle, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
    
    out.write(frame)


