# #!
# # テンソルと演算
import tempfile
import time

import numpy as np
import tensorflow as tf

# ## テンソル
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))
# -
x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)

# ### NumPy互換性
ndarray = np.ones([3, 3])

print("TensorFlow演算によりnumpy配列は自動的にテンソルに変換される")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("またNumPy演算によりテンソルは自動的にnumpy配列に変換される")
print(np.add(tensor, 1))

print(".numpy()メソッドによりテンソルは明示的にnumpy配列に変換される")
print(tensor.numpy())

# ## GPU による高速化
x = tf.random.uniform([3, 3])

print("利用できるGPUはあるか: ")
print(tf.config.experimental.list_physical_devices("GPU"))

print("テンソルはGPU #0にあるか:  ")
print(x.device.endswith("GPU:0"))

# ### 明示的デバイス配置


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start

    print("10 loops: {:0.2f}ms".format(1000 * result))


# !CPUでの実行を強制
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)
# !GPU #0があればその上での実行を強制
if tf.config.experimental.list_physical_devices("GPU"):
    print("On GPU:")
    with tf.device("GPU:0"):  # 2番めのGPUなら GPU:1, 3番目なら GPU:2 など
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
# ## データセット
# ### ソースDatasetの作成
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# !CSVファイルを作成
_, filename = tempfile.mkstemp()

with open(filename, "w") as f:
    f.write("Line 1\nLine 2\nLine 3\n")
ds_file = tf.data.TextLineDataset(filename)

# ### 変換の適用
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# ### イテレート
print("ds_tensors の要素:")
for x in ds_tensors:
    print(x)
print("\nds_file の要素:")
for x in ds_file:
    print(x)
