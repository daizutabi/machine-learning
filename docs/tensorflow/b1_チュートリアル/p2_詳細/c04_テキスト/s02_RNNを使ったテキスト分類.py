# # RNNを使ったテキスト分
# # 類(https://www.tensorflow.org/tutorials/text/text_classification_rnn)
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string], "")
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])


# ## Setup input pipeline
dataset, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]
encoder = info.features["text"].encoder
print("Vocabulary size: {}".format(encoder.vocab_size))
# -
sample_string = "Hello TensorFlow."
encoded_string = encoder.encode(sample_string)
print("Encoded string is {}".format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

# ## Prepare the data for training
BUFFER_SIZE = 10000
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

# ## Create the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)


# ## Train the model
history = model.fit(
    train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
)

# -
test_loss, test_acc = model.evaluate(test_dataset)

print("Test Loss: {}".format(test_loss))
print("Test Accuracy: {}".format(test_acc))
# -
plot_graphs(history, "accuracy")
# -
plot_graphs(history, "loss")


# ## Stack two or more LSTM layers
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
)
# -
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
# -
plot_graphs(history, 'accuracy')
# -
plot_graphs(history, 'loss')
