# # テキスト
import os

import tensorflow as tf
import tensorflow_datasets as tfds

DIRECTORY_URL = "https://storage.googleapis.com/download.tensorflow.org/data/illiad/"
FILE_NAMES = ["cowper.txt", "derby.txt", "butler.txt"]

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)
parent_dir = os.path.dirname(text_dir)


# ## テキストをデータセットに読み込む
def labeler(example, index):
    return example, tf.cast(index, tf.int64)


labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)
# -
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

for ex in all_labeled_data.take(5):
    print(ex)
# ## テキスト行を数字にエンコードする
# ### ボキャブラリーの構築
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()  # type: ignore
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)
vocab_size = len(vocabulary_set)
vocab_size
text_tensor
# ### サンプルをエンコードする
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


all_encoded_data = all_labeled_data.map(encode_map_fn)

# ## データセットを、テスト用と訓練用のバッチに分割する
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

# !（ゼロをパディングに使用した）新しいトークン番号を1つ導入したので、
# !ボキャブラリーサイズは1つ増えています。
vocab_size += 1

# ## モデルを構築する
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# !1 つ以上の Dense 層
# !`for` 行の中のリストを編集して、層のサイズの実験をしてください
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
# !出力層 最初の引数はラベルの数
model.add(tf.keras.layers.Dense(3, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# ## モデルを訓練する
model.fit(train_data, epochs=3, validation_data=test_data)
# -
eval_loss, eval_acc = model.evaluate(test_data)

print("\nEval loss: {}, Eval accuracy: {}".format(eval_loss, eval_acc))
