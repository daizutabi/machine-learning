# #!
# # CSV
import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# !numpy の値を読みやすくする
np.set_printoptions(precision=3, suppress=True)

# ## データのロード
# !入力ファイル中の CSV 列
with open(train_file_path, "r") as f:
    names_row = f.readline()
CSV_COLUMNS = names_row.rstrip("\n").split(",")
LABELS = [0, 1]
LABEL_COLUMN = "survived"
FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]


# -
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,  # 見やすく表示するために意図して小さく設定しています
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
    )
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
# -
examples, labels = next(iter(raw_train_data))  # 最初のバッチのみ
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)
# ## データの前処理
# ### カテゴリデータ
CATEGORIES = {
    "sex": ["male", "female"],
    "class": ["First", "Second", "Third"],
    "deck": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "embark_town": ["Cherbourg", "Southhampton", "Queenstown"],
    "alone": ["y", "n"],
}


# -
def process_categorical_data(data, categories):
    """カテゴリ値を表すワンホット・エンコーディングされたテンソルを返す"""

    # 最初の ' ' を取り除く
    data = tf.strings.regex_replace(data, "^ ", "")
    # 最後の '.' を取り除く
    data = tf.strings.regex_replace(data, r"\.$", "")

    # ワンホット・エンコーディング
    # data を1次元（リスト）から2次元（要素が1個のリストのリスト）にリシェープ
    data = tf.reshape(data, [-1, 1])
    # それぞれの要素について、カテゴリ数の長さの真偽値のリストで、
    # 要素とカテゴリのラベルが一致したところが True となるものを作成
    data = categories == data
    # 真偽値を浮動小数点数にキャスト
    data = tf.cast(data, tf.float32)

    # エンコーディング全体を次の1行に収めることもできる：
    # data = tf.cast(categories == tf.reshape(data, [-1, 1]), tf.float32)
    return data


# ### 連続データ
def process_continuous_data(data, mean):
    # data の標準化
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    return tf.reshape(data, [-1, 1])


MEANS = {
    "age": 29.631308,
    "n_siblings_spouses": 0.545455,
    "parch": 0.379585,
    "fare": 34.385399,
}


# ### データの前処理
def preprocess(features, labels):

    # カテゴリ特徴量の処理
    for feature in CATEGORIES.keys():
        features[feature] = process_categorical_data(
            features[feature], CATEGORIES[feature]
        )
    # 連続特徴量の処理
    for feature in MEANS.keys():
        features[feature] = process_continuous_data(features[feature], MEANS[feature])
    # 特徴量を1つのテンソルに組み立てる
    features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)

    return features, labels


train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)


# ## モデルの構築
def get_model(input_dim, hidden_units=[100]):
    """複数の層を持つ Keras モデルを作成

  引数:
    input_dim: (int) バッチ中のアイテムの形状
    labels_dim: (int) ラベルの形状
    hidden_units: [int] DNN の層のサイズ（入力層が先）
    learning_rate: (float) オプティマイザの学習率

  戻り値:
    Keras モデル
  """

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    return model


input_shape, output_shape = train_data.output_shapes
input_dimension = input_shape.dims[1]  # [0] はバッチサイズ

# ## 訓練、評価、そして予測
model = get_model(input_dimension)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_data, epochs=20)
# -
test_loss, test_accuracy = model.evaluate(test_data)
print("\n\nTest Loss {}, Test Accuracy {}".format(test_loss, test_accuracy))
# -
predictions = model.predict(test_data)
# !結果のいくつかを表示
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print(
        "Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"),
    )
