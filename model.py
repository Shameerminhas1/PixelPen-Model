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



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.applications import VGG16
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

# # Load the VGG16 model, excluding the top dense layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# # Freeze the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom top layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# # Define the new model
# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.0001),  # Fine-tuning with a lower learning rate
#     metrics=['accuracy']
# )

# model.summary()

# # Set Up Callbacks
# callbacks = [
#     EarlyStopping(patience=10, restore_best_weights=True),
#     ModelCheckpoint('best_model.keras', save_best_only=True)
# ]

# # Initial Training
# history = model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=validation_generator,
#     callbacks=callbacks
# )

# # Evaluate the initial model
# val_loss, val_accuracy = model.evaluate(validation_generator)
# print(f"Initial validation loss: {val_loss}")
# print(f"Initial validation accuracy: {val_accuracy}")

# # Plot Initial Training History
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

# # Unfreeze some layers of the base model for fine-tuning
# for layer in base_model.layers[-4:]:  # Unfreezing last 4 layers as an example
#     layer.trainable = True

# # Recompile the model with a lower learning rate for fine-tuning
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=1e-5),
#     metrics=['accuracy']
# )

# # Fine-tune the model
# fine_tune_history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=validation_generator,
#     callbacks=callbacks
# )

# # Evaluate the fine-tuned model
# val_loss, val_accuracy = model.evaluate(validation_generator)
# print(f"Validation loss after fine-tuning: {val_loss}")
# print(f"Validation accuracy after fine-tuning: {val_accuracy}")

# # Plot Fine-Tuning History
# plt.plot(fine_tune_history.history['accuracy'], label='train accuracy')
# plt.plot(fine_tune_history.history['val_accuracy'], label='val accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# plt.plot(fine_tune_history.history['loss'], label='train loss')
# plt.plot(fine_tune_history.history['val_loss'], label='val loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#######################################################################################33

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.applications import VGG16
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

# # Load the VGG16 model, excluding the top dense layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# # Freeze the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom top layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# # Define the new model
# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.0001),  # Fine-tuning with a lower learning rate
#     metrics=['accuracy']
# )

# model.summary()

# # Set Up Callbacks
# callbacks = [
#     EarlyStopping(patience=10, restore_best_weights=True),
#     ModelCheckpoint('best_model.keras', save_best_only=True)
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




import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Bidirectional, LSTM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subfolder = os.path.join(folder, subdir)
        if os.path.isdir(subfolder):
            for filename in os.listdir(subfolder):
                img_path = os.path.join(subfolder, filename)
                if img_path.endswith(".png"):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 32))
                    images.append(img)
                    labels.append(subdir)
    return images, labels

# Function to preprocess data
def preprocess_data(train_folder, val_folder):
    train_images, train_labels = load_images_from_folder(train_folder)
    val_images, val_labels = load_images_from_folder(val_folder)
    
    train_images = np.array(train_images).reshape(-1, 32, 128, 1) / 255.0
    val_images = np.array(val_images).reshape(-1, 32, 128, 1) / 255.0
    
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    val_labels = le.transform(val_labels)
    
    return train_images, to_categorical(train_labels), val_images, to_categorical(val_labels), le.classes_

# Paths to train and validation folders
train_folder = 'D:/PixelPen-Model/dataset/train'
val_folder = 'D:/PixelPen-Model/dataset/validation'

# Preprocess the data
X_train, y_train, X_val, y_val, classes = preprocess_data(train_folder, val_folder)

# Function to create CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Input shape and number of classes
input_shape = (32, 128, 1)
num_classes = len(classes)

# Create, summarize, and train CNN model
cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.summary()
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
cnn_model.save('cnn_model.h5')

# Flatten the data for KNN
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Create, train, and evaluate KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_flat, y_train.argmax(axis=1))
y_pred = knn_model.predict(X_val_flat)
print(f"KNN Accuracy: {accuracy_score(y_val.argmax(axis=1), y_pred)}")

# Function to create RNN model
def create_rnn_model(input_shape, num_classes):
    model = Sequential([
        Reshape(target_shape=(32, 128), input_shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create, summarize, and train RNN model
rnn_model = create_rnn_model(input_shape, num_classes)
rnn_model.summary()
rnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
rnn_model.save('rnn_model.h5')
