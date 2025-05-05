#!/usr/bin/env python
"""
Utility script to download YOLO models
"""
import argparse
from ultralytics import YOLO

def download_model(model_name):
    """
    Download a YOLO model from Ultralytics
    
    Args:
        model_name (str): Name of the model to download (e.g., 'yolov8n', 'yolo11s')
    """
    print(f"Downloading {model_name} model...")
    try:
        # This will download the model if it's not already cached
        model = YOLO(f"{model_name}.pt")
        print(f"Successfully downloaded {model_name} model")
        print(f"Model path: {model.ckpt_path}")
        return model
    except Exception as e:
        print(f"Error downloading {model_name} model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download YOLO models from Ultralytics")
    parser.add_argument("--model", default="yolo11s", help="Model name to download (default: yolo11s)")
    
    args = parser.parse_args()
    download_model(args.model)

if __name__ == "__main__":
    main()