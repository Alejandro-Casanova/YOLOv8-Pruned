@echo off
setlocal enabledelayedexpansion

REM Iterate over all .onnx files in my_models folder
for %%f in ("my_models\*.onnx") do (
    echo Processing model: %%f
    python .\test_val_model.py -m %%f
)

endlocal