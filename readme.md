Sure! Here’s a complete **ready-to-copy README.md** file for your project:

````markdown
# Infectious vs Non-Infectious Material Classifier

This project uses a trained Keras/TensorFlow model to classify materials captured by your webcam as **infectious** or **non-infectious** in real time.

## Requirements

- Python 3.10+  
- TensorFlow / Keras  
- OpenCV  
- NumPy  

Install dependencies:

```bash
pip install -r requirements.txt
````

## Files

* `keras_Model.h5` → Your trained Keras model
* `labels.txt` → Class names, one per line (`infectious` and `non_infectious`)
* `webcam_classify.py` → Script for real-time webcam classification

## Usage

1. Connect a webcam to your computer.
2. Run the script:

```bash
python webcam_classify.py
```

3. The webcam window will show the live video feed with predicted class and confidence.
4. Press `ESC` to exit.

## Notes for Developers

* The model expects images resized to **224x224 pixels** and normalized to **\[-1, 1]**.
* `labels.txt` must match the order of classes used in training.
* You can modify `webcam_classify.py` to change the input size if your model uses a different resolution.
* To use a new model, replace `keras_Model.h5` and update `labels.txt` to match the new model's classes.

