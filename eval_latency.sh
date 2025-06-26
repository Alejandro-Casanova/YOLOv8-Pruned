#!/bin/bash

# Iterate over all .onnx files in my_models folder
for f in my_models/*.onnx; do
    echo "Processing model: $f"
    for seed in {1..5}; do
        echo "  Using seed: $seed"
        python ./test_val_model.py -m "$f" -s "$seed"
    done
done
