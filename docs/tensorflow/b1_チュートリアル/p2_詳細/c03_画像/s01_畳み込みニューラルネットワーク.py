# #!

# # 畳み込みニューラルネットワーク
# # (https://www.tensorflow.org/tutorials/images/cnn)

from tensorflow.keras import datasets, layers, models

# gpu = tf.config.experimental.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)

# ## MNISTデータセットのダウンロードと準備
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# !ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0

# ## 畳み込みの基礎部分の作成
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))


# ## 上にDenseレイヤーを追加
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()


# ## モデルのコンパイルと学習
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=5)

# ## モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
test_acc
