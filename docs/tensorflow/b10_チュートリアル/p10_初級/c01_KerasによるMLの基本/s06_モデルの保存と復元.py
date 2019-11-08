# # モデルの保存と復元
# ## 設定
import os
import shutil

import tensorflow as tf
from tensorflow import keras

# ### サンプルデータセットの取得
dataset = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = dataset

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# ### モデルの定義
# !短いシーケンシャルモデルを返す関数
def create_model():
    model = tf.keras.models.Sequential(
        [
            keras.layers.Dense(512, activation="relu", input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# !基本的なモデルのインスタンスを作る
model = create_model()
model.summary()

# ## 訓練中にチェックポイントを保存する
# ### チェックポイントコールバックの使い方
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_weights_only=True, verbose=1
)

model = create_model()

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback],
)
# -
model = create_model()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# -
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
shutil.rmtree(checkpoint_dir)

# ### チェックポイントコールバックのオプション
# !ファイル名に(`str.format`を使って)エポック数を埋め込みます
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=True,
    # 重みを5エポックごとに保存します
    period=5,
)

model = create_model()
model.fit(
    train_images,
    train_labels,
    epochs=50,
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0,
)

# -
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# -
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
shutil.rmtree(checkpoint_dir)
