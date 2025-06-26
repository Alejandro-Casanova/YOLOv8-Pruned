import json
import os
import sys
import onnxruntime as ort
import numpy as np
import time
import statistics
from tqdm import tqdm
from torch.autograd import profiler
import argparse

parser = argparse.ArgumentParser(description="Evaluate ONNX model latency.")
parser.add_argument('-m', '--model', type=str, default="my_models/best_00_imgsz320.onnx", help="Path to ONNX model")
parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

onnx_model_path = args.model
np.random.seed(args.seed)

# Extract model class from suffix (e.g., imgsz320_quant8)
a = onnx_model_path.split('.')[-2]
b = a.split('_')
img_size = int(b[3].replace('imgsz', '')) if 'imgsz' in b[3] else 640  # Default to 640 if not specified
model_class = b[3] + '_' + b[4] if len(b) > 4 else b[3] 
prune_rate = b[2]

# Load ONNX model
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], enable_profiling=True)

# Get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = np.float32  

# Create dummy input (batch size 1, 3 channels, 640x640 image)
dummy_input = np.random.randn(1, 3, img_size, img_size).astype(input_dtype)


# sys.exit()
# Warm-up runs
print("Warming up the model...")
for _ in range(20):
    _ = session.run(None, {input_name: dummy_input})

# Multiple iterations
num_iterations = 100
inference_times = []
for _ in tqdm(range(num_iterations), desc="Running inference"):
    start_time = time.time()
    _ = session.run(None, {input_name: dummy_input})
    end_time = time.time()
    inference_times.append(end_time - start_time)

average_inference_time_s = statistics.mean(inference_times)
print(f"Average inference time: {average_inference_time_s:.6f} seconds")
print(f"FPS: {1 / average_inference_time_s:.2f}")

results = {
    'inference speed (ms)': float(average_inference_time_s * 1e3),
    'FPS': float(1 / average_inference_time_s),
    'model_class': model_class,
    'seed': args.seed,
}

os.makedirs(f'my_models/export_results/{model_class}/{prune_rate}', exist_ok=True)
with open(f'my_models/export_results/{model_class}/{prune_rate}/{args.seed}.json', 'w') as f:
    json.dump(results, f, indent=4)
sys.exit()