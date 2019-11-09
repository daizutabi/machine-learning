# # カスタム訓練：基本
import matplotlib.pyplot as plt
import tensorflow as tf

# ## 変数
# !Python の状態を使う
x = tf.zeros([3, 3])
x += 2  # これは x = x + 2 と等価で, x の元の値を変えているわけではない
print(x)
# -
v = tf.Variable(1.0)
# !Python の `assert` を条件をテストするデバッグ文として使用
assert v.numpy() == 1.0

# !`v` に値を再代入
v.assign(3.0)
assert v.numpy() == 3.0

# !`v` に TensorFlow の `tf.square()` 演算を適用し再代入
v.assign(tf.square(v))
assert v.numpy() == 9.0


# ## 線形モデルの適合
# ### モデルの定義
class Model:
    def __init__(self):
        # 重みを `5.0` に、バイアスを  `0.0` に初期化
        # 実際には、これらの値は乱数で初期化するべき（例えば `tf.random.normal` を使って）
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()

assert model(3.0).numpy() == 15.0


# ### 損失関数の定義
def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.square(predicted_y - target_y))


# ### 訓練データの取得
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

plt.scatter(inputs, outputs, c="b")
plt.scatter(inputs, model(inputs), c="r")

print("Current loss: %1.6f" % loss(model(inputs), outputs).numpy())


# ### 訓練ループの定義
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


# -
model = Model()

# !後ほどプロットするために、W 値と b 値の履歴を集める
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print(
        "Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" % (epoch, Ws[-1], bs[-1], current_loss)
    )

# !すべてをプロット
plt.plot(epochs, Ws, "r", epochs, bs, "b")
plt.plot([TRUE_W] * len(epochs), "r--", [TRUE_b] * len(epochs), "b--")
plt.legend(["W", "b", "True W", "True b"])
