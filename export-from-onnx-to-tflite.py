import onnx
import tensorflow as tf
# import tf2onnx
import numpy as np
import os
import onnx
# import tf2onnx
import tensorflow as tf
import shutil
import sys
import argparse
import onnx
import onnx_tf

def onnx_to_tflite(onnx_model_path, tflite_model_path):
    # Load ONNX model

    onnx_model = onnx.load(onnx_model_path)
    print(f"Loaded ONNX model from {onnx_model_path}")
    temporal_dir = onnx_model_path.replace('.onnx', '.pb')
    onnx_tf.converter.convert(onnx_model, output_path=temporal_dir)

    # make a converter object from the saved tensorflow file
    converter = tf.lite.TFLiteConverter.from_saved_model(temporal_dir)
    # tell converter which type of optimization techniques to use
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # to view the best option for optimization read documentation of tflite about optimization
    # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

    # convert the model 
    tf_lite_model = converter.convert()
    # save the converted model 
    open(tflite_model_path, 'wb').write(tf_lite_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to TFLite format.")
    parser.add_argument('-i', "--input_onnx", help="Path to the input ONNX model file.")
    parser.add_argument('-o', "--output_tflite", help="Path to the output TFLite model file.")
    args = parser.parse_args()

    onnx_to_tflite(args.input_onnx, args.output_tflite)
    print(f"Converted {args.input_onnx} to {args.output_tflite}")
