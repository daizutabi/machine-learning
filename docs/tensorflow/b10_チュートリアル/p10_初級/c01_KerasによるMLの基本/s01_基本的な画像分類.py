# #!
# # 基本的な画像の分類

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.__version__

# ## ファッションMNISTデータセットのロード
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# ## データの観察
train_images.shape
# -
len(train_labels)
# -
train_labels
# -
test_images.shape

# ## データの前処理
plt.imshow(train_images[0])
plt.colorbar()
# -
train_images = train_images / 255.0
test_images = test_images / 255.0
# -
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# ## モデルの構築
# ### 層の設定
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
# ### モデルのコンパイル
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# ## モデルの訓練
model.fit(train_images, train_labels, epochs=5)

# ## 正解率の評価
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# ## 予測する
predictions = model.predict(test_images)
predictions[0]
# -
(np.argmax(predictions[0]), test_labels[0])
