# webcam_classify_savedmodel.py
import cv2
import numpy as np
from keras.layers import TFSMLayer
from keras import Input, Model

# -----------------------------
# Load the legacy SavedModel
# -----------------------------
model_layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")

# Wrap in a Keras Model
input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)
outputs = model_layer(inputs)
model = Model(inputs, outputs)

# -----------------------------
# Load class names
# -----------------------------
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

# -----------------------------
# Start webcam
# -----------------------------
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Resize image for the model
    img_resized = cv2.resize(frame, (224, 224))

    # Preprocess: normalize to [-1, 1]
    img_array = np.expand_dims(img_resized.astype(np.float32)/127.5 - 1, axis=0)

    # Predict
    prediction_dict = model.predict(img_array)
    prediction_values = list(prediction_dict.values())[0][0]
    index = np.argmax(prediction_values)
    class_name = class_names[index]
    confidence = prediction_values[index]

    # Display prediction on console
    print(f"Class: {class_name}, Confidence: {confidence*100:.2f}%")

    # Display webcam feed with prediction
    cv2.putText(frame, f"{class_name} ({confidence*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    # Exit on 'Q' or 'q' key
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

camera.release()
cv2.destroyAllWindows()
