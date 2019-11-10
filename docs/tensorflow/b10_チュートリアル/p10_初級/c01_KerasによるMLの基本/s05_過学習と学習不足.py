# # 過学習と学習不足
# # (https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# ## IMDBデータセットのダウンロード
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(
    num_words=NUM_WORDS
)


def multi_hot_sequences(sequences, dimension):
    # 形状が (len(sequences), dimension)ですべて0の行列を作る
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # 特定のインデックスに対してresults[i] を１に設定する
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# -
plt.plot(train_data[0])

# ## 過学習のデモ
# ### 比較基準を作る
baseline_model = keras.Sequential(
    [
        # `.summary` を見るために`input_shape`が必要
        keras.layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

baseline_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

baseline_model.summary()
# -
baseline_history = baseline_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)
# ### より小さいモデルの構築
smaller_model = keras.Sequential(
    [
        keras.layers.Dense(4, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

smaller_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

smaller_model.summary()
# -
smaller_history = smaller_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)

# ### より大きなモデルの構築
bigger_model = keras.models.Sequential(
    [
        keras.layers.Dense(512, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

bigger_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

bigger_model.summary()
# -
bigger_history = bigger_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)


# ### 訓練時と検証時の損失をグラフにする
def plot_history(histories, key="binary_crossentropy"):
    plt.figure(figsize=(12, 8))

    for name, history in histories:
        val = plt.plot(
            history.epoch,
            history.history["val_" + key],
            "--",
            label=name.title() + " Val",
        )
        plt.plot(
            history.epoch,
            history.history[key],
            color=val[0].get_color(),
            label=name.title() + " Train",
        )
    plt.xlabel("Epochs")
    plt.ylabel(key.replace("_", " ").title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


plot_history(
    [
        ("baseline", baseline_history),
        ("smaller", smaller_history),
        ("bigger", bigger_history),
    ]
)

# ## 過学習防止の戦略
# ### 重みの正則化を加える
l2_model = keras.models.Sequential(
    [
        keras.layers.Dense(
            16,
            kernel_regularizer=keras.regularizers.l2(0.001),
            activation="relu",
            input_shape=(NUM_WORDS,),
        ),
        keras.layers.Dense(
            16, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu"
        ),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

l2_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

l2_model_history = l2_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)
# -
plot_history([("baseline", baseline_history), ("l2", l2_model_history)])


# ### ドロップアウトを追加する
dpt_model = keras.models.Sequential(
    [
        keras.layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

dpt_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

dpt_model_history = dpt_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)

# -
plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])
