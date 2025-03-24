import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model('models/traffic_sign_model_40kmh.h5')

test_dir = 'dataset/test'

IMG_HEIGHT, IMG_WIDTH = 224, 224

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(img_batch)[0][0]
    label = "Biển 40km/h" if prediction > 0.5 else "Không phải biển 40km/h"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return img, label, confidence

plt.figure(figsize=(15, 10))

speed_sign_dir = os.path.join(test_dir, '40kmh')
other_dir = os.path.join(test_dir, 'other')

speed_sign_images = [os.path.join(speed_sign_dir, f) for f in os.listdir(speed_sign_dir)[:5]]
other_images = [os.path.join(other_dir, f) for f in os.listdir(other_dir)[:5]]

test_images = speed_sign_images + other_images
np.random.shuffle(test_images)

for i, image_path in enumerate(test_images[:10]):
    img, label, confidence = predict_image(image_path)
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"{label}\nConfidence: {confidence:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('models/test_predictions.png')
plt.show()

print("Kiểm tra mô hình hoàn tất!")