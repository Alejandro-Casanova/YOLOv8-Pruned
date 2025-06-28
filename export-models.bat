@echo off
setlocal enabledelayedexpansion

REM List of model suffixes
set models=00 10 20 30 50 60 70 80 90
set image_sizes=160 320 480 640

REM Iterate over each model and image size, export both normal and quantized models
for %%X in (%models%) do (
    for %%S in (%image_sizes%) do (
        REM Export normal model
        echo Exporting model best_%%X.pt with image size %%S...
        python .\export-model.py -m .\my-models\best_%%X.pt -i %%S -f onnx --optimize
        set "new_name=best_%%X_imgsz%%S.onnx"
        ren .\my-models\best_%%X.onnx !new_name!

        REM Export quantized model
        echo Exporting model best_%%X.pt with image size %%S and 8 bit quantization...
        python .\export-model.py -m .\my-models\best_%%X.pt -i %%S -f onnx --optimize -q
    )
)

echo All models exported.
pause