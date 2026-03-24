import argparse
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from utils import download_video


def run(video_path: str, output_path: Path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, FPS={fps}, Total frames={total_frames}")

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