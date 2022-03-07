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


if __name__ == '__main__':
    # estimate_constant()
    base()