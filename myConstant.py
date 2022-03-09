import tensorflow as tf
import numpy as np

def estimate_constant():
    cons = tf.constant(4)
    print(cons)

    cons2 = tf.constant([1,2,3,4])
    print(cons2)

    #通过使用 np.array 或 tensor.numpy 方法，可以将张量转换为 NumPy 数组
    print(cons2.numpy())
    print(np.array(cons2))

def base():
    a = tf.constant([[1,2],[3,4]])
    b = tf.constant([[10,20],[30,40]])

    # 逐元素相加 element-wise add
    print(tf.add(a,b))
    # 等价于element wise add
    print(a+b)
    # 逐元素相乘
    print(tf.multiply(a,b))
    # 等价于element wise相乘
    print(a * b)

    # 矩阵乘法
    print(tf.matmul(a,b))
    # 点乘，等价于上面的矩阵乘法
    print(a@b)

    # 找到最大的值
    print(tf.reduce_max(a))
    # 找到最大值的index
    print(tf.argmax(a))

def myIndex():

    # 单轴
    rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    print(rank_1_tensor.numpy())

    print("First:", rank_1_tensor[0].numpy())
    print("Second:", rank_1_tensor[1].numpy())
    print("Last:", rank_1_tensor[-1].numpy())

    print("每个元素:", rank_1_tensor[:].numpy())
    print("前四个元素:", rank_1_tensor[:4].numpy())
    print("第四个到最后的元素:", rank_1_tensor[4:].numpy())
    print("第二到七个元素:", rank_1_tensor[2:7].numpy())
    print("每%3的元素（每第三个元素）:", rank_1_tensor[::3].numpy())
    print("逆序:", rank_1_tensor[::-1].numpy())

    # 多轴（同单轴）
    rank_2_tensor = tf.constant([[1,2],[3,4]])
    print(rank_2_tensor)
    print(rank_2_tensor[1, 1])
    print(rank_2_tensor[1, 1].numpy())

    print("第二行的全部:", rank_2_tensor[1, :].numpy())
    print("第二列的全部:", rank_2_tensor[:, 1].numpy())
    print("最后一行:", rank_2_tensor[-1, :].numpy())
    print("First item in last column:", rank_2_tensor[0, -1].numpy())
    print("Skip the first row:")
    print(rank_2_tensor[1:, :].numpy(), "\n")

def reshape():
    var_x = tf.Variable(tf.constant([[1], [2], [3]]))
    print(var_x.shape)
    reshaped = tf.reshape(var_x, [1, 3])
    print(reshaped.shape)
    print(tf.reshape(reshaped,[-1]))

def broadcast():
    x = tf.constant([1, 2, 3])

    y = tf.constant(2)
    z = tf.constant([2, 2, 2])
    # 下面三种方法等效
    print(tf.multiply(x, 2))
    print(x * y)
    print(x * z)

    x = tf.reshape(x, [3, 1])
    # tf.range 是生成一个x1~x2的序列，下面的意思就是从1开始生成到5，左闭右开，1234
    y = tf.range(1, 5)
    print(x, "\n")
    print(y, "\n")
    print(tf.multiply(x, y))
    # 下面是不使用广播的实现方式，耗费内存
    x_stretch = tf.constant([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3]])
    y_stretch = tf.constant([[1, 2, 3, 4],
                             [1, 2, 3, 4],
                             [1, 2, 3, 4]])
    print(x_stretch * y_stretch)

    # tf.broadcast_to，把一个array广播到指定的shape，但是会创建新的内存对象，消耗内存
    print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))

def ragged_tensor():
    ragged_list = [
        [0,1,2,3],
        [4,5],
        [6,7,8],
        [9]
    ]
    ragged_tensor = tf.ragged.constant(ragged_list)
    print(ragged_tensor)

def string_tensor():
    tensor_of_strings = tf.constant(["Gray wolf",
                                     "Quick brown fox",
                                     "Lazy dog"])
    print(tensor_of_strings)

    unicode_tensor = tf.constant(u"啦啦aa")
    print(unicode_tensor)

    # 字符串切割
    print(tf.strings.split(tensor_of_strings, sep=" "))

    # 字符串转数字
    number_of_strings = tf.constant("1 10 100")
    print(tf.strings.to_number(tf.strings.split(number_of_strings)))

def sparse_tensor():
    dense_tensor = tf.constant([
        [1,0,0,0],
        [0,0,2,0],
        [0,0,0,0]
    ])
    # 稠密张量转稀疏张量没有直接的方法，需要自己构造
    dense_idx = tf.where(tf.not_equal(dense_tensor,0))
    # tf.gather根据index取数
    sparse_tensor = tf.sparse.SparseTensor(indices=dense_idx,values=tf.gather_nd(dense_tensor,dense_idx),dense_shape=dense_tensor.get_shape())
    print(sparse_tensor)

    sparse_tensor_demo = tf.sparse.SparseTensor(indices=[[0,0],[1,2]],values=[1,2],dense_shape=[3,4])
    print(sparse_tensor_demo)

    #稀疏张量转稠密张量
    print(tf.sparse.to_dense(sparse_tensor))
    print(tf.sparse.to_dense(sparse_tensor_demo))

if __name__ == '__main__':
    # estimate_constant()
    # base()
    # myIndex()
    # reshape()
    # broadcast()
    # ragged_tensor()
    # string_tensor()
    sparse_tensor()