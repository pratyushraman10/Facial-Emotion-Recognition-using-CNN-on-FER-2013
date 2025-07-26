FER-2013 Emotion Recognition using Convolutional Neural Networks

This project implements and trains a Convolutional Neural Network (CNN) to classify facial emotions based on the FER-2013 dataset. The model processes grayscale facial images to predict one of seven distinct emotion categories.

Dataset Description

The FER-2013 dataset consists of 48x48 pixel grayscale images of faces, each labeled with one of seven emotion categories:

Label | Emotion
------|---------
0     | Angry
1     | Disgust
2     | Fear
3     | Happy
4     | Sad
5     | Surprise
6     | Neutral

The dataset is provided as a CSV file (fer2013.csv) containing at least the following columns:

- emotion: Integer label representing the emotion category.
- pixels: String of space-separated pixel values for each image.

Please ensure that the fer2013.csv file is available in the working directory when running the scripts.

Overview of Implementation

1. Required Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

These libraries facilitate data manipulation (numpy, pandas), model construction and training (TensorFlow with Keras), and the splitting of data into subsets (scikit-learn).

2. Data Loading and Preprocessing

df = pd.read_csv("fer2013.csv")
df_clean = df[df['pixels'].str.split().str.len() == 48*48]

- Loads the raw dataset.
- Cleans the data by retaining only rows with exactly 2304 pixel values (48 x 48), ensuring consistency in input image size.

3. Feature Construction

X = np.vstack(df_clean['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).to_numpy())
X = X.reshape(-1, 48, 48, 1) / 255.0

- Converts pixel data from strings to numerical arrays.
- Reshapes the data to 4D tensors compatible with CNN input requirements (samples, height, width, channels).
- Normalizes pixel intensities to the [0,1] range for improved training stability.

4. Label Encoding

y = pd.get_dummies(df_clean['emotion']).reindex(columns=range(7), fill_value=0).values

- Converts categorical emotion labels into one-hot encoded vectors suitable for multi-class classification.

5. Dataset Splitting

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_clean['emotion']
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=np.argmax(y_train, axis=1)
)

- Divides the dataset into training, validation, and test subsets.
- Maintains class balance across splits using stratification.

6. Model Architecture

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

- The CNN model consists of multiple convolutional layers with ReLU activations for effective pattern recognition.
- MaxPooling layers downsample feature maps, reducing dimensionality while preserving important features.
- Fully connected (Dense) layers perform high-level reasoning before the final output layer.
- The output layer uses softmax activation for probability distribution over the seven emotion classes.
- The model is compiled with the Adam optimizer and categorical crossentropy loss, targeting classification accuracy.

7. Training Procedure

model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val)
)

- Trains the model over 20 epochs with mini-batches of size 64.
- Uses validation data for monitoring potential overfitting and generalization during training.

8. Evaluation

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

- Evaluates the final model performance on the unseen test data.
- Reports accuracy as the key performance metric.

Additional Remarks

- This baseline model is intended as a starting point for facial emotion recognition.
- Improvements can be achieved via hyperparameter tuning, advanced architectures, and data augmentation techniques.
- Proper handling of imbalanced classes and more rigorous validation strategies are recommended for real-world applications.

Please feel free to contact the project maintainer for further guidance or collaboration opportunities.
