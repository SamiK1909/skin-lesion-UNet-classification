from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dropout, UpSampling2D, Input, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dropout, UpSampling2D, Input
import json
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
metadata_train = pd.read_csv('metadataTrain.csv')
train_dir = os.path.join('Train', 'Train')
train_images = [os.path.join(train_dir, x)
                for x in os.listdir(train_dir) if x.endswith('.jpg')]
train_segs = [os.path.join(train_dir, x)
              for x in os.listdir(train_dir) if x.endswith('_seg.png')]

metadata_train['SEX'].fillna(metadata_train['SEX'].mode()[0], inplace=True)
metadata_train['AGE'].fillna(metadata_train['AGE'].median(), inplace=True)
metadata_train['POSITION'].fillna(
    metadata_train['POSITION'].mode()[0], inplace=True)

# Define paths and check existence
train_dir = 'Train/Train'
metadata_train['image_path'] = metadata_train['ID'].apply(
    lambda id: os.path.join(train_dir, f"{id}.jpg"))
metadata_train['seg_path'] = metadata_train['ID'].apply(
    lambda id: os.path.join(train_dir, f"{id}_seg.png"))
metadata_train['image_exists'] = metadata_train['image_path'].apply(
    os.path.exists)
metadata_train['seg_exists'] = metadata_train['seg_path'].apply(os.path.exists)

# Filter data to ensure image files exist
metadata_train = metadata_train[metadata_train['image_exists']]

# Optionally create a version of the DataFrame with only entries where segmentation masks exist
metadata_train_with_segs = metadata_train[metadata_train['seg_exists']]

# Assuming metadata_train is your DataFrame and you have a column 'CLASS' for labels
train_data, val_data = train_test_split(
    metadata_train, test_size=0.2, random_state=42, stratify=metadata_train['CLASS'])


def load_and_preprocess_image(path, size=(256, 256)):
    img = load_img(path, target_size=size, color_mode="rgb")
    img = img_to_array(img) / 255.0
    return img


def load_and_preprocess_mask(path, size=(256, 256)):
    mask = load_img(path, target_size=size, color_mode="grayscale")
    mask = img_to_array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)  # Binary mask
    return mask


# def get_unet(input_shape=(256, 256, 3)):
#     inputs = Input(input_shape)

#     # Contracting Path
#     c1 = Conv2D(16, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(inputs)
#     c1 = Dropout(0.1)(c1)
#     c1 = Conv2D(16, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)

#     c2 = Conv2D(32, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(p1)
#     c2 = Dropout(0.1)(c2)
#     c2 = Conv2D(32, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)

#     # Bottleneck
#     c3 = Conv2D(64, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(p2)
#     c3 = Dropout(0.2)(c3)
#     c3 = Conv2D(64, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(c3)

#     # Expansive Path
#     u6 = UpSampling2D((2, 2))(c3)
#     u6 = concatenate([u6, c2])
#     c6 = Conv2D(32, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(u6)
#     c6 = Dropout(0.2)(c6)
#     c6 = Conv2D(32, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(c6)

#     u7 = UpSampling2D((2, 2))(c6)
#     u7 = concatenate([u7, c1])
#     c7 = Conv2D(16, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(u7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv2D(16, (3, 3), activation='relu',
#                 kernel_initializer='he_normal', padding='same')(c7)

#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

#     model = Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer='adam', loss='binary_crossentropy',
#                   metrics=['accuracy'])

#     return model

# # Create U-Net model
# model = get_unet()


def get_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    # Contracting Path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Expanding Path
    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


model_new = get_unet()

# Create the U-Net model
model_new = get_unet(input_shape=(256, 256, 3))

# Compile the model
model_new.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

data_with_masks = metadata_train[metadata_train['seg_exists']]
x = np.array([load_and_preprocess_image(path)
             for path in data_with_masks['image_path']])
y = np.array([load_and_preprocess_mask(path)
             for path in data_with_masks['seg_path']])

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=data_with_masks['CLASS'], random_state=42)

results = model_new.fit(x_train, y_train, batch_size=32,
                        epochs=50, validation_data=(x_val, y_val))

model_new.save('gpu_unet_improved.h5')

# Convert the history.history dict to a JSON file
with open('gpu_unet_improved.json', 'w') as f:
    json.dump(results.history, f)
