# # 映画レビューのテキスト分類

import altair as alt
import pandas as pd
from tensorflow import keras

# ## IMDB datasetのダウンロード
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# ## データの観察
print(f"Training entries: {len(train_data)}, labels: {len(train_labels)}")
print(train_data[0])
# -
# !レビューごとに長さが異なる。
len(train_data[0]), len(train_data[1])

# ### 整数を単語に戻してみる
# !単語を整数にマッピングする辞書
word_index = imdb.get_word_index()

# !インデックスの最初の方は予約済み
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


decode_review(train_data[0])
# ## データの準備
# ! 長さを標準化する
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=256
)
len(train_data[0]), len(train_data[1])

# ## モデルの構築
# !入力の形式は映画レビューで使われている語彙数（10,000語）
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()

# ### 損失関数とオプティマイザ
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ## 検証用データを作る
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# ## モデルの訓練
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1,
)

# ## モデルの評価
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

# ## 正解率と損失の時系列グラフを描く
df = pd.DataFrame(history.history)
df.index.name = "epoch"
df.reset_index(inplace=True)
df = pd.melt(df, id_vars=["epoch"], value_vars=["accuracy", "val_accuracy"])
chart = alt.Chart(df).mark_line(point=True).properties(width=200, height=150)
chart.encode(
    x="epoch", y=alt.Y("value", scale=alt.Scale(domain=[0.5, 1])), color="variable"
)
