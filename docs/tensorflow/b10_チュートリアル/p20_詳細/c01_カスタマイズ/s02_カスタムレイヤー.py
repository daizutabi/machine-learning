# # カスタムレイヤー

import tensorflow as tf

# ## レイヤー：有用な演算の共通セット
# !tf.keras.layers パッケージの中では、レイヤーはオブジェクトです。
# !レイヤーを構築するためにすることは、単にオブジェクトを作成するだけです。
# !ほとんどのレイヤーでは、最初の引数が出力の次元あるいはチャネル数を表します。
layer = tf.keras.layers.Dense(100)
# !入力の次元数は多くの場合不要となっています。それは、レイヤーが最初に使われる際に
# !推定可能だからです。ただし、引数として渡すことで手動で指定することも可能です。
# !これは複雑なモデルを構築する場合に役に立つでしょう。
layer = tf.keras.layers.Dense(3, input_shape=(None, 4))
# !レイヤーを使うには、単純にcallします。
layer(tf.zeros([5, 4]))
# -
# !レイヤーにはたくさんの便利なメソッドがあります。例えば、`layer.variables`を使って
# !レイヤーのすべての変数を調べることができます。訓練可能な変数は、 `layer.trainable_variables`
# !でわかります。この例では、全結合レイヤーには重みとバイアスの変数があります。
layer.variables
# !これらの変数には便利なアクセサを使ってアクセス可能です。
layer.kernel, layer.bias


# ## カスタムレイヤーの実装
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), self.num_outputs]
        )

    def call(self, input):
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(4)
print(layer(tf.zeros([3, 5])))
print(layer.trainable_variables)


# ## モデル：レイヤーの組み合わせ
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name="")
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
with tf.device("CPU:0"):
    print(block(tf.zeros([1, 2, 3, 3])))
    print([x.name for x in block.trainable_variables])
# -
my_seq = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(2, 1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(3, (1, 1)),
        tf.keras.layers.BatchNormalization(),
    ]
)

with tf.device("CPU:0"):
    my_seq(tf.zeros([1, 2, 3, 3]))
