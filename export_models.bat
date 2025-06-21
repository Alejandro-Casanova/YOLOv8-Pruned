@echo off
setlocal enabledelayedexpansion

REM List of model suffixes
set models=10 20 30 50 60 70 80 90

REM Iterate over each model and call export_model.py
for %%X in (%models%) do (
    echo Exporting model best_%%X.pt...
    python .\export_model.py -m .\my_models\best_%%X.pt -i 320 -f onnx --optimize

    REM Rename the exported ONNX file to add imgsz suffix
    set "new_name=best_%%X_imgsz320.onnx"
    ren .\my_models\best_%%X.onnx !new_name!
)

REM Do the same but apply 16 bit quantization
for %%X in (%models%) do (
    echo Exporting model best_%%X.pt with 16 bit quantization...
    python .\export_model.py -m .\my_models\best_%%X.pt -i 320 -f onnx --optimize -q16

    REM Rename the exported ONNX file to add imgsz suffix
    set "new_name=best_%%X_imgsz320_quant16.onnx"
    ren .\my_models\best_%%X.onnx !new_name!
)

REM Do the same but apply 8 bit quantization
for %%X in (%models%) do (
    echo Exporting model best_%%X.pt with 8 bit quantization...
    python .\export_model.py -m .\my_models\best_%%X.pt -i 320 -f onnx --optimize -q8

    REM Rename the exported ONNX file to add imgsz suffix
    set "new_name=best_%%X_imgsz320_quant8.onnx"
    ren .\my_models\best_%%X.onnx !new_name!
)

REM Do the same but for the 640 image size
for %%X in (%models%) do (
    echo Exporting model best_%%X.pt with image size 640...
    python .\export_model.py -m .\my_models\best_%%X.pt -i 640 -f onnx --optimize

    REM Rename the exported ONNX file to add imgsz suffix
    set "new_name=best_%%X_imgsz640.onnx"
    ren .\my_models\best_%%X.onnx !new_name!
)

REM Do the same for the unpruned model trash_yolov8n_pretrained.pt
echo Exporting unpruned model trash_yolov8n_pretrained.pt...
python .\export_model.py -m .\my_models\trash_yolov8n_pretrained.pt -i 320 -f onnx --optimize
ren .\my_models\trash_yolov8n_pretrained.onnx trash_yolov8n_pretrained_imgsz320.onnx
python .\export_model.py -m .\my_models\trash_yolov8n_pretrained.pt -i 640 -f onnx --optimize
ren .\my_models\trash_yolov8n_pretrained.onnx trash_yolov8n_pretrained_imgsz640.onnx
python .\export_model.py -m .\my_models\trash_yolov8n_pretrained.pt -i 320 -f onnx --optimize -q16
ren .\my_models\trash_yolov8n_pretrained.onnx trash_yolov8n_pretrained_imgsz320_quant16.onnx
python .\export_model.py -m .\my_models\trash_yolov8n_pretrained.pt -i 320 -f onnx --optimize -q8
ren .\my_models\trash_yolov8n_pretrained.onnx trash_yolov8n_pretrained_imgsz320_quant8.onnx

echo All models exported.
pause