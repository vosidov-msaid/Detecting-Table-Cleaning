import argparse
import os
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

from utils import download_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table cleaning detection")
    parser.add_argument("--video", type=str, 
                        default=None, help="Path to video file")
    parser.add_argument("--output", type=str, 
                        default="output.csv", help="Path to output file")
    args = parser.parse_args()
    
    if args.video is None:
        print("Error: Video file is required")
        exit(1)
    
    if not os.path.exists(args.video):
        download_video("https://drive.google.com/uc?id=1rYmJB13vvV96JuDUrBvlEXtoKFPWo75A", args.video)
    