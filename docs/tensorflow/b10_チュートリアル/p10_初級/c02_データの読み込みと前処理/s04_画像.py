# # 画像
import os
import pathlib
import random

import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# ## データセットのダウンロードと検査
# ### 画像の取得
url = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/flower_photos.tgz"
)
data_root_orig = tf.keras.utils.get_file(origin=url, fname="flower_photos", untar=True)
data_root_orig
data_root = pathlib.Path(data_root_orig)

for item in data_root.iterdir():
    print(os.path.basename(item))
# -
all_image_paths_ = list(data_root.glob("*/*"))
all_image_paths = [str(path) for path in all_image_paths_]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
image_count

# ## 画像の検査
for n in range(3):
    image_path = random.choice(all_image_paths)
    display.display(display.Image(image_path))
# ### 各画像のラベルの決定
label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
label_names
# -
label_to_index = dict((name, index) for index, name in enumerate(label_names))
label_to_index
# -
all_image_labels = [
    label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths
]

print("First 10 labels indices: ", all_image_labels[:10])


# ### 画像の読み込みと整形
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# ## tf.data.Datasetの構築
# ### 画像のデータセット
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
path_ds
# -
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2, 2, n + 1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
# ### (image, label)のペアのデータセット
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(4):
    print(label_names[label.numpy()])
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
image_label_ds

# ### 基本的な訓練手法
BATCH_SIZE = 32

# !シャッフルバッファのサイズをデータセットとおなじに設定することで、データが完全にシャッフルされる
# !ようにできます。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# !`prefetch`を使うことで、モデルの訓練中にバックグラウンドでデータセットがバッチを取得できます。
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

# -
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds
