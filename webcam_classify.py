from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]

    print(f"Class: {class_name}, Confidence: {confidence*100:.2f}%")

    cv2.putText(frame, f"{class_name}: {confidence*100:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Image", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
