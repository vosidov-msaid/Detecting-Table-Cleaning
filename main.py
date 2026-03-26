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
from table_tracker import TableTracker

STATE_EMPTY = config.STATE_EMPTY
STATE_OCCUPIED = config.STATE_OCCUPIED
STATE_APPROACH = config.STATE_APPROACH

model = YOLO(config.MODEL)

def get_video_info(cap: cv2.VideoCapture):
    """Get video information"""

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total_frames

def resize_video(frame):
    """Resize frames to 1280x720"""
    max_w, max_h = 1280, 720
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return h, w, scale, frame

def get_roi_frame(cap: cv2.VideoCapture):
    """Get ROI coordinates"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame at frame 0")

    print("[INFO] Selected the table area: ")

    _, _, _, frame = resize_video(frame)
    roi_xywh = cv2.selectROI("Select the table area", frame, showCrosshair=True, fromCenter=False)
    roi = tuple(int(v) for v in roi_xywh)
    cv2.destroyAllWindows()
    return roi

def draw_frame(frame, roi, state, person_boxes, tracker: TableTracker):
    x, y, w, h = roi

    color = {
        STATE_EMPTY: config.COLOR_EMPTY,
        STATE_OCCUPIED: config.COLOR_OCCUPIED,
        STATE_APPROACH: config.COLOR_APPROACH,
    }.get(state, (255, 255, 255))

    # ROI Frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    label = f"TABLE: {state}"
    cv2.rectangle(frame, (x, y - 30), (x + len(label) * 12, y), color, -1)
    cv2.putText(frame, label, (x + 4, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    # Person boxes
    for (bx1, by1, bx2, by2) in person_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)

    info_lines = [
        f"Events: {len(tracker.events) - 1}",
    ]

    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return frame

def bbox_iou_with_roi(bx1, by1, bx2, by2, rx, ry, rw, rh) -> float:
    ix1 = max(bx1, rx)
    iy1 = max(by1, ry)
    ix2 = min(bx2, rx + rw)
    iy2 = min(by2, ry + rh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    roi_area = rw * rh
    return inter / roi_area

def detect_person(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect person on frames and return coordinates"""
    results = model(frame, 
                        conf=config.THRESHOLD,
                        classes=[0],
                        verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes

def run(video_path: str, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Video information
    fps, width, height, total_frames = get_video_info(cap)
    print(f"[INFO] Video: {width}x{height}, FPS={fps}, Total frames={total_frames}")

    # ROI Frame
    roi = get_roi_frame(cap)
    print("[INFO] ROI: ", roi)

    tracker  = TableTracker(roi, fps)

    out_path = str(output_path / "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (1280, 720))

    frame_no = 0
    last_boxes = []

    # Video handling... (click Q for quit)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # Resize video
        _, _, _, frame = resize_video(frame)

        if frame_no % config.SKIP_FRAMES == 0:  
            # Detect person on frame        
            last_boxes = detect_person(frame)
        
        # Check person in ROI
        rx, ry, rw, rh = roi
        person_in_roi = any(
            bbox_iou_with_roi(bx1, by1, bx2, by2, rx, ry, rw, rh) >= config.IOU_THRESHOLD
            for (bx1, by1, bx2, by2) in last_boxes
        )
        tracker.update(frame_no, person_in_roi)
        
        # Draw frame
        vis = draw_frame(frame.copy(), roi, tracker.state, last_boxes, tracker)

        writer.write(vis)

        cv2.imshow("Table Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Exit")
            break
    
    cap.release()
    writer.release()

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