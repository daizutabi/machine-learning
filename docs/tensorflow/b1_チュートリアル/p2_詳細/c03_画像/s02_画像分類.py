# # 画像分類
# # (https://www.tensorflow.org/tutorials/images/classification)

import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ## データの読み込み
_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

train_cats_dir = os.path.join(train_dir, "cats")  # 学習用の猫画像のディレクトリ
train_dogs_dir = os.path.join(train_dir, "dogs")  # 学習用の犬画像のディレクトリ
validation_cats_dir = os.path.join(validation_dir, "cats")  # 検証用の猫画像のディレクトリ
validation_dogs_dir = os.path.join(validation_dir, "dogs")  # 検証用の犬画像のディレクトリ

# ### データの理解
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("total training cat images:", num_cats_tr)
print("total training dog images:", num_dogs_tr)
print("total validation cat images:", num_cats_val)
print("total validation dog images:", num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# -
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# ## データの準備
train_image_generator = ImageDataGenerator(rescale=1.0 / 255)  # 学習データのジェネレータ
validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)  # 検証データのジェネレータ

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)
# -
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

# ### 学習用画像の可視化
sample_training_images, _ = next(train_data_gen)


# !この関数は、1行5列のグリッド形式で画像をプロットし、画像は各列に配置されます。
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()


plotImages(sample_training_images[:5])


# ## モデルの構築と学習
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
model = Sequential(
    [
        Conv2D(16, 3, padding="same", activation="relu", input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
# -
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
)
# ### 学習結果の可視化
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")


# ## データ拡張（Data augmentation）
# ### 水平反転の適用
image_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ### 画像のランダムな回転
image_gen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# ### ズームによるデータ拡張の適用
image_gen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ### すべてのデータ拡張を同時に利用する
image_gen_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5,
)

train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ### 検証データジェネレータの構築
image_gen_val = ImageDataGenerator(rescale=1.0 / 255)

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

# ## ドロップアウトを追加した新しいネットワークの構築
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
model_new = Sequential(
    [
        Conv2D(16, 3, padding="same", activation="relu", input_shape=input_shape),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# ### モデルのコンパイル
model_new.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_new.summary()

# ### モデルの学習
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
)


# ### モデルの可視化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
