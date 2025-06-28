@echo off
setlocal enabledelayedexpansion

REM Iterate over all .onnx files in my-models folder
for %%f in ("my-models\*.onnx") do (
    echo Processing model: %%f
    python .\test-val-model.py -m %%f
)

endlocal