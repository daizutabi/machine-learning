# # 回帰：燃費を予測する
# # (https://www.tensorflow.org/tutorials/keras/regression)

import altair as alt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ## Auto MPG データセット
# ### データの取得
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/auto-mpg/auto-mpg.data"
)
dataset_path = keras.utils.get_file("auto-mpg.data", url,)
# -
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]
raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)
dataset = raw_dataset.copy()
dataset.tail()

# ### データのクレンジング
dataset.isna().sum()
# -
# !簡単のために削除
dataset = dataset.dropna()
# -
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0
dataset.tail()
# ### データを訓練用セットとテスト用セットに分割
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ### データの観察
sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
)
# -
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# ### ラベルと特徴量の分離
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# ### データの正規化
def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# ## モデル
# ### モデルの構築
def build_model():
    layer = layers.Dense(64, activation="relu", input_shape=[len(train_dataset.keys())])
    model = keras.Sequential(
        [layer, layers.Dense(64, activation="relu"), layers.Dense(1)]
    )

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


model = build_model()

# ### モデルの検証
model.summary()


# ### モデルの訓練
# !エポックが終わるごとにドットを一つ出力することで進捗を表示
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()],
)
# -
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()


# -
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    df = pd.melt(hist, id_vars=["epoch"], value_vars=["mae", "val_mae"])
    chart = alt.Chart(df).mark_line(clip=True).properties(width=200, height=150)
    y = alt.Y("value", scale=alt.Scale(domain=[0, 5]))
    return chart.encode(x="epoch", y=y, color="variable")


plot_history(history)
# -
model = build_model()

# !patienceは改善が見られるかを監視するエポック数を表すパラメーター
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()],
)

plot_history(history)
# -
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
# ### モデルを使った予測
test_predictions = model.predict(normed_test_data).flatten()
df = pd.DataFrame({"label": test_labels, "pred": test_predictions})
scale = alt.Scale(domain=[0, 50])
x = alt.X("label", title="True Value [MPG]", scale=scale)
y = alt.Y("pred", title="Predictions [MPG]", scale=scale)
chart = alt.Chart(df).mark_point().encode(x=x, y=y)
chart.properties(width=200, height=200)
