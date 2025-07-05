@echo off
setlocal enabledelayedexpansion

REM Iterate over all .tf files 

for %%f in ("my-models\best_10_imgsz160_tflite\*.tflite") do (
    echo Processing model: %%f
    python .\test-val-model.py -m %%f -o .\my-models\tflite-results
)

for %%f in ("my-models\best_10_imgsz320_tflite\*.tflite") do (
    echo Processing model: %%f
    python .\test-val-model.py -m %%f -o .\my-models\tflite-results
)

for %%f in ("my-models\best_10_imgsz480_tflite\*.tflite") do (
    echo Processing model: %%f
    python .\test-val-model.py -m %%f -o .\my-models\tflite-results
)

endlocal