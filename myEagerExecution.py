import tensorflow as tf
import numpy as np

tf.executing_eagerly()

def testEager():
    # test eager
    x = [[2.]]
    m = tf.matmul(x, x)
    print("hello, {}".format(m))

    a = tf.constant([[1,2],
                     [3,4]])

    b = tf.add(a,1)
    print(b)

    c = tf.multiply(a,b)
    print(c)
    print(a*b)
    print(np.multiply(a,b))
    print(c.numpy())

def testGradientTape():
    w = tf.Variable([[1.0]])
    with tf.GradientTape() as tape:
        loss = w * w

    grad = tape.gradient(loss, w)
    print(grad)

def testTraining():
    pass

if __name__ == '__main__':
    # testEager()
    testGradientTape()