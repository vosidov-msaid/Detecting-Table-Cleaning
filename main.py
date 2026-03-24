import argparse
import os
from tracemalloc import Frame
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

import config
from utils import download_video

STATE_EMPTY    = "EMPTY"
STATE_OCCUPIED = "OCCUPIED"
STATE_APPROACH = "APPROACH"

def get_video_info(cap: cv2.VideoCapture):
    """Get video information"""

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total_frames

def resize_video(frame):
    max_w, max_h = 1280, 720
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    return h, w, scale, frame

def get_rio_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[INFO] Selected the table area: ")

    h, w, scale, frame = resize_video(frame)

    roi_xywh = cv2.selectROI("Select the table area", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    return roi_xywh

def draw_frame(frame, roi, state, person_boxes):
    x, y, w, h = roi

    color = {
        STATE_EMPTY: "EMPTY",
        STATE_OCCUPIED: "OCCUPIED",
        STATE_APPROACH: "APPROACH",
    }.get(state, (255, 255, 255))

    # RIO Frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Person boxes
    for (bx1, by1, bx2, by2) in person_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)

    return frame

def detect_person(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    model = YOLO(config.model)
    results = model(frame, 
                        conf=config.threshold,
                        classes=[0],
                        verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes

def run(video_path: str, output_path: Path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Video information
    fps, width, height, total_frames = get_video_info(cap)
    print(f"[INFO] Video: {width}x{height}, FPS={fps}, Total frames={total_frames}")

    # RIO Frame
    roi = get_rio_frame(cap)
    print("[INFO] RIO: ", roi)

    # Video handling... (click Q for quit)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, scale, frame = resize_video(frame)

        last_boxes = detect_person(frame)

        # Draw frame
        vis = draw_frame(frame, roi, "STATE_OCCUPIED", last_boxes)
        cv2.imshow("Table Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Прерывание пользователем.")
            break
    cap.release()
    cv2.destroyAllWindows()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table cleaning detection")
    parser.add_argument("--video", type=str, 
                        default=None, help="Path to video file")
    parser.add_argument("--output", type=str, 
                        default="output", help="Path to output directory")
    args = parser.parse_args()
    
    if args.video is None:
        print("Error: Video file is required")
        exit(1)
    
    if not os.path.exists(args.video):
        download_video("https://drive.google.com/uc?id=1rYmJB13vvV96JuDUrBvlEXtoKFPWo75A", args.video)
    
    run(args.video, Path(args.output))