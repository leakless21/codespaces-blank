import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def convert_pt_to_onnx(model_path, output_path=None, simplify=True):
    """
    Convert Ultralytics YOLO model (.pt) to ONNX format
    
    Args:
        model_path (str): Path to the .pt model
        output_path (str, optional): Path to save the ONNX model. If None, will use same name with .onnx extension
        simplify (bool): Whether to simplify the ONNX model
        
    Returns:
        str: Path to the converted ONNX model
    """
    # Load the model
    model = YOLO(model_path)
    
    # Determine output path if not provided
    if output_path is None:
        output_path = Path(model_path).with_suffix('.onnx')
    
    # Export the model to ONNX format
    success = model.export(format='onnx', simplify=simplify)
    
    if success:
        print(f"Successfully converted {model_path} to ONNX format")
        print(f"ONNX model saved to: {output_path}")
        return str(output_path)
    else:
        print(f"Failed to convert {model_path} to ONNX format")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO models from .pt to .onnx format")
    parser.add_argument("--model", required=True, help="Path to the .pt model file")
    parser.add_argument("--output", help="Path to save the ONNX model (default: same name with .onnx extension)")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify", 
                        help="Disable ONNX model simplification")
    
    args = parser.parse_args()
    
    convert_pt_to_onnx(args.model, args.output, args.simplify)

if __name__ == "__main__":
    main()