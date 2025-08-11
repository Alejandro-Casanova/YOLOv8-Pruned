# Ultralytics YOLOv8 Pruning

This repository provides tools for pruning YOLOv8 models.

Based on [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.

## Setup

Tested for Python v3.11.

**Create a virtual environment and install dependencies:**

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  pip install -r requirements-venv.txt
  ```

## Usage

**Run the pruning script:**

  ```bash
  python yolo8-pruning.py
  ```

## Notes

- Ensure you have Python installed.
- Modify `yolo8-pruning.py` as needed for your specific pruning configuration.
- For more details, refer to the script's documentation or comments.
