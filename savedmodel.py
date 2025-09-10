import cv2
import numpy as np
from keras.layers import TFSMLayer
from keras import Input, Model


model_layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")

input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)
outputs = model_layer(inputs)
model = Model(inputs, outputs)


with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]


camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    img_resized = cv2.resize(frame, (224, 224))

    img_array = np.expand_dims(img_resized.astype(np.float32)/127.5 - 1, axis=0)

    prediction_dict = model.predict(img_array)
    prediction_values = list(prediction_dict.values())[0][0]
    index = np.argmax(prediction_values)
    class_name = class_names[index]
    confidence = prediction_values[index]

    print(f"Class: {class_name}, Confidence: {confidence*100:.2f}%")

    cv2.putText(frame, f"{class_name} ({confidence*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

camera.release()
cv2.destroyAllWindows()
