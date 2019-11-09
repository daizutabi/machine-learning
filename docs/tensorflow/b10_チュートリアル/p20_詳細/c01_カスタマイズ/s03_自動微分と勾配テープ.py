# # 自動微分と勾配テープ
import tensorflow as tf

# ## 勾配テープ
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
# !元の入力テンソル x に対する z の微分
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0
# -
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
# !テープを使って中間値 y に対する z の微分を計算
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0
# -
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y
print(t.gradient(z, x))  # 108.0 (4*x^3 at x = 3)
print(t.gradient(y, x))  # 6.0
del t  # テープへの参照を削除


# ### 制御フローの記録
def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output


def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)


x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

# ### 高次勾配
x = tf.Variable(1.0)  # 1.0 で初期化された TensorFlow 変数を作成

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # ’t’ コンテキストマネジャー内で勾配を計算
    # これは勾配計算も同様に微分可能であるということ
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
