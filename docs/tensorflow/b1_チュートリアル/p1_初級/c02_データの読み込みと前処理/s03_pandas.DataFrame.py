# # pandas.DataFrame
# # (https://www.tensorflow.org/tutorials/load_data/pandas_dataframe)

import pandas as pd
import tensorflow as tf

url = "https://storage.googleapis.com/applied-dl/heart.csv"
csv_file = tf.keras.utils.get_file("heart.csv", url)
df = pd.read_csv(csv_file)
df.head()
# -
df["thal"] = pd.Categorical(df["thal"])
df["thal"] = df.thal.cat.codes
df.head()

# ## tf.data.Datasetを使ってDataFrameをロード
target = df.pop("target")
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
    print("Features: {}, Target: {}".format(feat, targ))
# -
train_dataset = dataset.shuffle(len(df)).batch(1)


# ## モデルの構築と訓練
def get_compiled_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)
