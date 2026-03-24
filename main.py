import argparse
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


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
        