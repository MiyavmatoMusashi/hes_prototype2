import cv2
import numpy as np
from tensorflow.keras.models import train_model

CLASSES = ['+', '-', '=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
model = train_model("char_model.h5")


def preprocess_and_segment(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    positions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char = thresh[y:y + h, x:x + w]
        char = cv2.resize(char, (28, 28))
        chars.append(char)
        positions.append(x)

    chars = [x for _, x in sorted(zip(positions, chars))]
    return chars


def predict_equation(chars):
    equation = ''
    for char in chars:
        input_img = char.reshape(1, 28, 28, 1) / 255.0
        pred = model.predict(input_img)
        idx = np.argmax(pred)
        equation += CLASSES[idx]
    return equation


def check_equation(eq_str):
    try:
        left, right = eq_str.split('=')
        return eval(left) == eval(right)
    except Exception as e:
        print("Hata:", e)
        return False


chars = preprocess_and_segment("denklem1.png")
eq_str = predict_equation(chars)
print("Tanınan Denklem:", eq_str)
print("Doğru mu?", check_equation(eq_str))
