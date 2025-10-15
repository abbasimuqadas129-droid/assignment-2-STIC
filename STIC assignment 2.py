# ======================================================
# 🌿 Simple CNN for Leaf Disease Classification (Assignment 2 - Baseline Model)
# ======================================================

import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. Fix Randomness (so all students get same results) ---
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()

# --- 2. Load Dataset ---
train_data = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/train",
    image_size=(128, 128),
    batch_size=32,
    label_mode='int',
    shuffle=True,
    seed=seed
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode='int',
    shuffle=False
)

# Store class names before mapping
class_names = train_data.class_names


# --- 3. Normalize Images (0–255 → 0–1) ---
train_data = train_data.map(lambda x, y: (x / 255.0, y))
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# --- 4. Build CNN Model ---
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3, seed=seed),
    layers.Dense(len(class_names), activation='softmax') # Use the stored class_names
])

# --- 5. Compile Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Show Model Summary ---
print("\n📊 Model Summary:")
model.summary()

# --- 7. Train Model (with Timing) ---
print("\n🚀 Training CNN Model...")
start_time = time.time()

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    verbose=1
)

end_time = time.time()
training_time = (end_time - start_time) / 60
print(f"\n⏱️ Training Time: {training_time:.2f} minutes")

# --- 8. Evaluate Model ---
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"✅ Final Test Accuracy: {test_acc:.4f}")

# --- 9. Plot Accuracy ---
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Simple CNN - Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# --- 10. Show Class Names ---
print("\n🌿 Classes Detected:", class_names)