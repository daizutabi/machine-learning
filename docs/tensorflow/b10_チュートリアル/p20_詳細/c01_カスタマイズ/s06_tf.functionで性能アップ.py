# # tf.functionで性能アップ
import contextlib

import tensorflow as tf


# !遭遇するかもしれないいくつかのエラーをデモするためのヘルパー関数
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print("Caught expected exception \n  {}: {}".format(error_class, e))
    except Exception as e:
        print("Got unexpected exception \n  {}: {}".format(type(e), e))
    else:
        raise Exception(
            "Expected {} to be raised but no error was raised!".format(error_class)
        )


# -
# !function は演算のように振る舞う
@tf.function
def add(a, b):
    return a + b


add(tf.ones([2, 2]), tf.ones([2, 2]))  # [[2., 2.], [2., 2.]]

# -
# !function は勾配を計算できる
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
tape.gradient(result, v)


# -
# !function 内で function を使うこともできる
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)


dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))


# ## トレーシングとポリモーフィズム
# !Function はポリモーフィック
@tf.function
def double(a):
    print("Tracing with", a)
    return a + a


print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()
# -
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(
    tf.TensorSpec(shape=None, dtype=tf.string)
)
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))
# -
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(tf.equal(x % 2, 0), x // 2, 3 * x + 1)


print(next_collatz(tf.constant([1, 2])))
# !1次元のテンソルを input signature として指定しているので、これは失敗する
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))
# ## 引数はPythonか？Tensorか？


def train_one_step():
    pass


@tf.function
def train(num_steps):
    print("Tracing with num_steps = {}".format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()


train(num_steps=10)
train(num_steps=20)
# -
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))


# ## tf.function の中の副作用
@tf.function
def f(x):
    print("Traced with", x)
    tf.print("Executed with", x)


f(1)
f(1)
f(2)
# -
external_list = []


def side_effect(x):
    print("Python side effect")
    external_list.append(x)


@tf.function
def f2(x):
    tf.py_function(side_effect, inp=[x], Tout=[])


f2(1)
f2(1)
f2(1)
assert len(external_list) == 3
# !.numpy() call required because py_function casts 1 to tf.constant(1)
assert external_list[0].numpy() == 1


# ## Python の状態に注意
external_var = tf.Variable(0)


@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("Value of external_var:", external_var)


iterator = iter([0, 1, 2, 3])
buggy_consume_next(iterator)
# !次のコードは、イテレーターの次の値を使うのではなく、最初の値を再利用する
buggy_consume_next(iterator)
buggy_consume_next(iterator)


# -
def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print(
        "{}({}) contains {} nodes in its graph".format(
            f.__name__, ", ".join(map(str, args)), len(g.as_graph_def().node)
        )
    )


@tf.function  # type: ignore
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x)  # ダミー計算
    return loss


small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

generator = tf.data.Dataset.from_generator
measure_graph_size(train, generator(lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, generator(lambda: big_data, (tf.int32, tf.int32)))


# ## 自動的な依存関係の制御
# !自動的な依存関係の制御
a = tf.Variable(1.0)
b = tf.Variable(2.0)


@tf.function  # type:ignore
def f(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b


f(1.0, 2.0)  # 10.0


# ## 変数
@tf.function  # type:ignore
def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v


with assert_raises(ValueError):
    f(1.0)
# -
# !しかし、曖昧さの無いコードは大丈夫
v = tf.Variable(1.0)


@tf.function  # type:ignore
def f(x):
    return v.assign_add(x)


print(f(1.0))  # 2.0
print(f(2.0))  # 4.0

# -
# !初めて関数が実行されるときだけ変数が生成されることを保証できれば
# !tf.function 内で変数を作成できる


class C:
    pass


obj = C()
obj.v = None  # type:ignore


@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)


print(g(1.0))  # 2.0
print(g(2.0))  # 4.0
# -
# !変数の初期化は、関数の引数や他の変数の値に依存可能
# !制御の依存関係を生成するのと同じ手法で、正しい初期化の順序を発見可能
state = []  # type: ignore


@tf.function
def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))
    return state[0] * x * state[1]


print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))


# ## AutoGraphの使用
# !単純な繰り返し


@tf.function  # type: ignore
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


f(tf.random.uniform([5]))


# ### 条件分岐
def test_tf_cond(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == "cond" for node in g.as_graph_def().node):
        print("{}({}) uses tf.cond.".format(f.__name__, ", ".join(map(str, args))))
    else:
        print("{}({}) executes normally.".format(f.__name__, ", ".join(map(str, args))))


@tf.function
def hyperparam_cond(x, training=True):
    if training:
        x = tf.nn.dropout(x, rate=0.5)
    return x


@tf.function
def maybe_tensor_cond(x):
    if x < 0:
        x = -x
    return x


test_tf_cond(hyperparam_cond, tf.ones([1], dtype=tf.float32))
test_tf_cond(maybe_tensor_cond, tf.constant(-1))
test_tf_cond(maybe_tensor_cond, -1)


# ### 繰り返し
def test_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == "while" for node in g.as_graph_def().node):
        print(
            "{}({}) uses tf.while_loop.".format(f.__name__, ", ".join(map(str, args)))
        )
    elif any(node.name == "ReduceDataset" for node in g.as_graph_def().node):
        print(
            "{}({}) uses tf.data.Dataset.reduce.".format(
                f.__name__, ", ".join(map(str, args))
            )
        )
    else:
        print("{}({}) gets unrolled.".format(f.__name__, ", ".join(map(str, args))))


@tf.function
def for_in_range():
    x = 0
    for i in range(5):
        x += i
    return x


test_dynamically_unrolled(for_in_range)


# -
@tf.function
def for_in_tfrange():
    x = tf.constant(0, dtype=tf.int32)
    for i in tf.range(5):
        x += i
    return x


test_dynamically_unrolled(for_in_tfrange)


# -
@tf.function
def for_in_tfdataset():
    x = tf.constant(0, dtype=tf.int64)
    for i in tf.data.Dataset.range(5):
        x += i
    return x


test_dynamically_unrolled(for_in_tfdataset)


# -
@tf.function
def while_py_cond():
    x = 5
    while x > 0:
        x -= 1
    return x


test_dynamically_unrolled(while_py_cond)


# -
@tf.function
def while_tf_cond():
    x = tf.constant(5)
    while x > 0:
        x -= 1
    return x


test_dynamically_unrolled(while_tf_cond)

# -
@tf.function
def while_py_true_py_break(x):
    while True:  # py true
        if x == 0:  # py break
            break
        x -= 1
    return x


test_dynamically_unrolled(while_py_true_py_break, 5)


# -
@tf.function
def buggy_while_py_true_tf_break(x):
    while True:  # py true
        if tf.equal(x, 0):  # tf break
            break
        x -= 1
    return x


with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)
# -
@tf.function
def while_tf_true_tf_break(x):
    while tf.constant(True):  # tf true
        if x == 0:  # py break
            break
        x -= 1
    return x


test_dynamically_unrolled(while_tf_true_tf_break, 5)
# -
@tf.function
def buggy_py_for_tf_break():
    x = 0
    for i in range(5):  # py for
        if tf.equal(i, 3):  # tf break
            break
        x += i
    return x


with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_py_for_tf_break)
# -
@tf.function
def tf_for_py_break():
    x = 0
    for i in tf.range(5):  # tf for
        if i == 3:  # py break
            break
        x += i
    return x


test_dynamically_unrolled(tf_for_py_break)
