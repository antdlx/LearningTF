import tensorflow as tf

def estimate():
    con1 = tf.constant([1,2,3])
    var1 = tf.Variable(con1)
    print(var1)

    var2 = tf.Variable([2,3,4])
    print(var2)
    print(var2.numpy)
    # tf.Variable.assign 用于给变量重新赋值，但是新的值和旧的值的shape应该相同。赋值不会创建新的内存，会复用当前的内存
    var2.assign(
        [6,6,6]
    )
    print(var2)

    # 如果使用变量创建变量，会创建新的内存，不会在调用的时候互相干扰
    var3 = tf.Variable([1,1,1])
    var4 = tf.Variable(var3)
    print(var3.assign_add([2,2,2]))
    print(var4)

    var5 = tf.Variable([1,2,3], name="Hello",trainable=False)
    print(var5)

    with tf.device('CPU:0'):
      a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.Variable([[1.0, 2.0, 3.0]])

    with tf.device('GPU:0'):
      # Element-wise multiply
      k = a * b

    print(k)

if __name__ == '__main__':
    estimate()