# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Set up directory paths
# train_dir = "D:/PixelPen-Model/dataset/train"
# validation_dir = "D:/PixelPen-Model/dataset/validation"

# # Data Preprocessing
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Confirm the number of classes
# num_classes = len(train_generator.class_indices)
# print(f"Number of classes in training data: {num_classes}")
# print(f"Number of classes in validation data: {len(validation_generator.class_indices)}")

# # Build the CNN Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')  # Adjust the number of classes here
# ])

# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.001),
#     metrics=['accuracy']
# )

# model.summary()

# # Set Up Callbacks
# callbacks = [
#     EarlyStopping(patience=10, restore_best_weights=True),
#     ModelCheckpoint('best_model.keras', save_best_only=True)  # Change the extension to .keras
# ]

# # Train the Model
# history = model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=validation_generator,
#     callbacks=callbacks
# )

# # Evaluate the Model
# val_loss, val_accuracy = model.evaluate(validation_generator)
# print(f"Validation loss: {val_loss}")
# print(f"Validation accuracy: {val_accuracy}")

# # Plot Training History
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
##-- code with 3-4 % acc and < loss


import os
import cv2
import pytesseract
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# Path to the dataset
DATA_PATH = 'D:/PixelPen-Model/dataset'

# Image size
IMG_SIZE = (128, 32)

# Character list for encoding labels
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;'- "
char_dict = {char: idx for idx, char in enumerate(char_list)}
num_classes = len(char_list)

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract-OCR installation path

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def encode_label(label):
    return [char_dict[char] for char in label]

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def load_data(directory):
    images = []
    labels = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(subdir, file)
                label = extract_text_from_image(image_path)
                
                image = preprocess_image(image_path)
                encoded_label = encode_label(label)
                
                images.append(image)
                labels.append(encoded_label)
    return np.array(images), labels

# Load training and validation data
X_train, y_train = load_data(os.path.join(DATA_PATH, 'train'))
X_val, y_val = load_data(os.path.join(DATA_PATH, 'validation'))

# Pad labels to the max length
max_label_length = max(max(len(label) for label in y_train), max(len(label) for label in y_val))
y_train_padded = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen=max_label_length, padding='post', value=len(char_list))
y_val_padded = tf.keras.preprocessing.sequence.pad_sequences(y_val, maxlen=max_label_length, padding='post', value=len(char_list))

# Define the model
def build_crnn(input_shape, num_classes):
    input_data = layers.Input(name='input', shape=input_shape, dtype='float32')

    # CNN layers
    cnn = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_data)
    cnn = layers.MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
    cnn = layers.MaxPooling2D(pool_size=(2, 2))(cnn)

    # Reshape for RNN
    rnn_input = layers.Reshape(target_shape=(-1, cnn.shape[2] * cnn.shape[3]))(cnn)

    # RNN layers
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn_input)
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn)

    # Dense layer with softmax
    y_pred = layers.Dense(num_classes, activation='softmax')(rnn)

    model = models.Model(inputs=input_data, outputs=y_pred)
    return model

# Input shape
input_shape = (IMG_SIZE[1], IMG_SIZE[0], 1)  # (height, width, channels)

# Build and compile model
model = build_crnn(input_shape, num_classes)
model.summary()

# CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Define the model with CTC loss
labels = layers.Input(name='the_labels', shape=[max_label_length], dtype='float32')
input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

y_pred = model.output

ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model = models.Model(inputs=[model.input, labels, input_length, label_length], outputs=ctc_loss)
model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})

# Prepare training data
input_length_train = np.ones((len(X_train), 1)) * (IMG_SIZE[0] // 2 - 2)
label_length_train = np.array([len(label) for label in y_train])

input_length_val = np.ones((len(X_val), 1)) * (IMG_SIZE[0] // 2 - 2)
label_length_val = np.array([len(label) for label in y_val])

# Dummy zero array for the loss function
output_dummy_train = np.zeros(len(X_train))
output_dummy_val = np.zeros(len(X_val))

# Train the model
model.fit(
    x=[X_train, y_train_padded, input_length_train, label_length_train], 
    y=output_dummy_train,
    validation_data=([X_val, y_val_padded, input_length_val, label_length_val], output_dummy_val),
    batch_size=32,
    epochs=10
)
