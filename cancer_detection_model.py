import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset
df = pd.read_csv('data/GroundTruth.csv')

# Load images
images = []
labels = df.iloc[:, 1:].values
image_ids = df["image"].values

for img_id in image_ids:
    path = f"data/images/{img_id}.jpg"
    image = cv2.imread(path)
    image = cv2.resize(image, (150, 200))
    image = image.astype("float32") / 255
    images.append(image)

# Convert to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split data
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=30)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(200, 150, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.40),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=32)

# Save model
model.save("cnn_model.h5")
print("Model saved as cnn_model.h5")