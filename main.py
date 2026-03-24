import argparse
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from utils import download_video

def get_video_info(cap: cv2.VideoCapture):
    """Get video information"""

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total_frames

def get_rio_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[INFO] Selected the table area: ")

    max_w, max_h = 1280, 720
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    roi_xywh = cv2.selectROI("Select the table area", frame, showCrosshair=True, fromCenter=False)
    roi_xywh = tuple(int(v * scale) for v in roi_xywh)
    cv2.destroyAllWindows()
    return roi_xywh

def run(video_path: str, output_path: Path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps, width, height, total_frames = get_video_info(cap)
    
    print(f"Video: {width}x{height}, FPS={fps}, Total frames={total_frames}")

    roi_xywh = get_rio_frame(cap)
    print(roi_xywh)

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