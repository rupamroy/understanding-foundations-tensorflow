import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
    with tf.Session() as sess:
        # y = Ax + b

        A = tf.constant([2.5, 4.0], name='var_W')

        x = tf.placeholder(tf.float32, name='x')
        b = tf.constant([5.0, 10.0], name='var_b')

        y = A * x + b

        print(sess.run(y, feed_dict={x: [10, 100]}))

        assert y.graph is g1

g2 = tf.Graph()

with g2.as_default():
    with tf.Session() as sess: 
        # y = A^x
         A = tf.constant([5, 7], tf.int32, name='var_A')
         x = tf.placeholder(tf.int32, name='x')

         y = tf.pow(A, x, name="y")

         print(sess.run(y, feed_dict={x: [3, 5]}))

         assert y.graph is g2