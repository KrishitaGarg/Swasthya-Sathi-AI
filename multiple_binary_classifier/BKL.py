import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('../data/GroundTruth.csv')

# Select images with BKL
dfBKL = df[df["BKL"] == 1][["image", "BKL"]]
dfNotBKL = df[df["BKL"] == 0][["image", "BKL"]]

# Balance the dataset by undersampling
dfNotBKL = dfNotBKL.sample(frac=0.04, random_state=42)
df = pd.concat([dfBKL, dfNotBKL]).sample(frac=1, random_state=42)

# Prepare image array
imageArray = df["image"].to_numpy()

# Load and preprocess images
Images = []
def processImage(image):
    image = cv2.resize(image, (150, 200))
    image = image.astype("float32") / 255
    return image

for img in imageArray:
    path = os.path.join("../data/images", img + ".jpg")
    image = cv2.imread(path)
    if image is not None:
        Images.append(processImage(image))

# Convert images to numpy array
Images = np.array(Images)
output = df["BKL"].to_numpy()

# Train-test split
xTrain, xTest, yTrain, yTest = train_test_split(Images, output, test_size=0.2, random_state=42)
xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.2, random_state=42)

# Model architecture
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
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=10, batch_size=32)

# Evaluate and predict
predictions = model.predict(xTest)
preds = [1 if p >= 0.5 else 0 for p in predictions]

# Classification report
report = classification_report(yTest, preds, target_names=["Not BKL", "BKL"])
print(report)

# Save model
model.save("BKL_model.h5")
