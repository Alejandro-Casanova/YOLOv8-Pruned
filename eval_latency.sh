#!/bin/bash

# Iterate over seeds first, then over all .onnx files in my_models folder
for seed in $(seq 1 5); do
    echo "Using seed: $seed"
    for f in my_models/*.onnx; do
        echo "  Processing model: $f"
        python ./eval_latency.py -m "$f" -s "$seed"
    done
done
