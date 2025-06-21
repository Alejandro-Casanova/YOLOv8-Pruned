from ultralytics import YOLO
import argparse
from yolov8_c2v_patch import C2f_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX format")
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('-f', '--format', type=str, default='tflite', help='Export format (tflite, onnx, coreml, engine)')
    parser.add_argument('-i', '--imgsz', type=int, default=640, help='Image size for export')
    parser.add_argument('--optimize', action='store_true', help='Optimize the model during export')
    parser.add_argument('-q8', '--quantize8', action='store_true', help='Quantize the model to 8-bit during export')
    parser.add_argument('-q16', '--quantize16', action='store_true', help='Quantize the model to 16-bit during export')

    args = parser.parse_args()

    # Check if both quantization flags are set
    if args.quantize8 and args.quantize16:
        raise ValueError("Cannot set both --quantize8 and --quantize16. Choose one.")
    
    # Load the YOLOv8 model
    model = YOLO(args.model)

    # Export the model
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        optimize=args.optimize,
        int8=args.quantize8,
        half=args.quantize16,
        simplify=args.optimize,
    )