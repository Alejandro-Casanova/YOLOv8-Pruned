#!/bin/bash

# Iterate over seeds first, then over all .onnx files in my-models folder
for seed in $(seq 1 5); do
    echo "Using seed: $seed"
    for f in my-models/*.onnx; do
        echo "  Processing model: $f"
        python ./eval-latency.py -m "$f" -s "$seed"
    done
done
