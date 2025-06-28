import sys
import cv2
from ultralytics import YOLO
import argparse
from yolov8_c2v_patch import C2f_v2
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
import torch
import torchvision as tv
import onnx
import onnxruntime as ort
from onnxruntime import quantization
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat
import cv2
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX format")
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('-f', '--format', type=str, default='tflite', help='Export format (tflite, onnx, coreml, engine)')
    parser.add_argument('-i', '--imgsz', type=int, default=640, help='Image size for export')
    parser.add_argument('--optimize', action='store_true', help='Optimize the model during export')
    parser.add_argument('-q', '--quantize', action='store_true', help='Quantize the model to 8-bit during export')
    parser.add_argument('--q-dynamic', action='store_true', help='Use dynamic quantization for ONNX export')

    args = parser.parse_args()
    
    # Load the YOLOv8 model
    model = YOLO(args.model)
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        optimize=args.optimize,
        int8=args.quantize,
        simplify=args.optimize,
        data="my-datasets/trash-detect.yaml"
    )
    
    if args.format == "onnx" and args.quantize:

        onnx_model_path = args.model.replace('.pt', '.onnx')
        onnx_model_prep_path = onnx_model_path.replace('.onnx', '_prep.onnx')
        q8_model_path = onnx_model_path.replace('.onnx', f"_imgsz{args.imgsz}_q8.onnx")

        quantization.shape_inference.quant_pre_process(onnx_model_path, onnx_model_prep_path, skip_symbolic_shape=False)

        if args.q_dynamic:
            # Use the exported ONNX model path
            quantize_dynamic(
                onnx_model_path,
                q8_model_path,
                weight_type=QuantType.QUInt8
            )
        else:
            class ImageCalibrationDataReader(CalibrationDataReader):
                def __init__(self, image_paths):
                    self.image_paths = image_paths
                    self.idx = 0
                    self.input_name = "images"

                def preprocess(self, frame):
                    frame = cv2.imread(frame)
                    X = cv2.resize(frame, (args.imgsz, args.imgsz))
                    image_data = np.array(X).astype(np.float32) / 255.0  # Normalize to [0, 1] range
                    image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
                    return image_data

                def get_next(self):
                    if self.idx >= len(self.image_paths):
                        return None

                    image_path = self.image_paths[self.idx]
                    input_data = self.preprocess(image_path)
                    self.idx += 1
                    return {self.input_name: input_data}
            calibration_image_paths = glob.glob("../datasets/trash_detection_optimal_split/images/val/*.jpg")
            if not calibration_image_paths:
                raise ValueError("No calibration images found for quantization.")

            calibration_data_reader = ImageCalibrationDataReader(calibration_image_paths)
            quantize_static(
                onnx_model_prep_path,
                q8_model_path,
                calibration_data_reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QUInt8,
                activation_type=QuantType.QUInt8,
                # per_channel=False,
                # reduce_range=True,
                nodes_to_exclude=[
                    '/model.22/Concat_5', '/model.22/Mul_2', '/model.22/Sigmoid',
                    '/model.22/Concat_4', '/model.22/Div_1', '/model.22/Add_2', '/model.22/Sub_1', 
                    '/model.22/Sub', '/model.22/Add1', '/model.22/Slice_1', '/model.22/Slice',
                    '/model.22/Concat_3', '/model.22/Split', '/model.22/Sigmoid',
                    '/model.22/dfl/Reshape', '/model.22/dfl/Transpose', '/model.22/dfl/Softmax', 
                    '/model.22/dfl/conv/Conv', '/model.22/dfl/Reshape_1', '/model.22/Slice_1',
                    '/model.22/Slice', '/model.22/Add_1', '/model.22/Sub', '/model.22/Div_1',
                    '/model.22/Concat_4', '/model.22/Mul_2', '/model.22/Concat_5',
                ],
                
            )
        print(f"8-bit quantized ONNX model saved to {q8_model_path}")
        if os.path.exists(onnx_model_prep_path):
            os.remove(onnx_model_prep_path)
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)