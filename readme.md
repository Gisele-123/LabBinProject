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
```

## Files

* `model.savedmodel` → Your trained TensorFlow SavedModel (from Teachable Machine)  
* `labels.txt` → Class names, one per line (`infectious` and `non_infectious`)  
* `webcam_classify_savedmodel.py` → Script for real-time webcam classification  

## Usage

1. Connect a webcam to your computer or make sure your laptop's camera works well.
2. Run the script:

```bash
python webcam_classify_savedmodel.py
```

3. The webcam window will show the live video feed with predicted class and confidence.
4. Press **`Q`** (or `q`) to exit the webcam feed.

## Notes for Developers

* The model expects images resized to **224x224 pixels** and normalized to **\[-1, 1]**.
* `labels.txt` must match the order of classes used in training.
* You can modify `webcam_classify_savedmodel.py` to change the input size if your model uses a different resolution.
* To use a new model, replace `model.savedmodel` and update `labels.txt` to match the new model's classes.
