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



if __name__ == '__main__':
    # estimate_constant()
    # base()
    # myIndex()
    reshape()