import sys
import onnxruntime as ort
import numpy as np
import time
import statistics
from tqdm import tqdm
from torch.autograd import profiler

# Load ONNX model
onnx_model_path = "my_models/best_00_imgsz320.onnx"  # Update with your ONNX model path
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], enable_profiling=True)

# Get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = np.float32  

# Create dummy input (batch size 1, 3 channels, 640x640 image)
dummy_input = np.random.randn(1, 3, 320, 320).astype(input_dtype)

# Note: torch.autograd.profiler is for PyTorch models, not ONNX Runtime.
# For ONNX Runtime, you can use built-in profiling as follows:

_ = session.run(None, {input_name: dummy_input})  # Run a single inference to generate some profiling data
profile_file = session.end_profiling()
print(f"ONNX Runtime profiling data saved to: {profile_file}")

sys.exit()
# Warm-up runs
for _ in range(100):
    _ = session.run(None, {input_name: dummy_input})

# Multiple iterations
num_iterations = 500
inference_times = []
for _ in tqdm(range(num_iterations), desc="Running inference"):
    start_time = time.time()
    _ = session.run(None, {input_name: dummy_input})
    end_time = time.time()
    inference_times.append(end_time - start_time)

average_inference_time = statistics.mean(inference_times)
print(f"Average inference time: {average_inference_time:.6f} seconds")
print(f"FPS: {1 / average_inference_time:.2f}")