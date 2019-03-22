import tensorflow as tf

# y = Wx + b

W = tf.constant([10, 100], name='const_W')

# these placeholders can be tensors of any shape as the sahe is not explicitly mentioned
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

# tf.multiply is element wise multiplicatiuon instead of matrix multiplication , W and X should be of the same rank and shape
Wx = tf.multiply(W,x, name='Wx')

y = tf.add(Wx, b, name='y')

#y_ = x -b

y_= tf.subtract(x, b, name="y_");

with tf.Session() as sess:
    print("Intermediate resulty: Wx=", sess.run(Wx, feed_dict={x: [3,33]}))
    print("Final result: Wx + b= ", sess.run(fetches=y, feed_dict={x: [5, 50], b: [7, 9]}))

    # Here we fecth two placeholders in the same sess.run
    print('Two results: [Wx+b, x-b]=',\
            sess.run(fetches=[x, y_], feed_dict={x: [5,50], b:[7, 9]}))

writer = tf.summary.FileWriter('./digging_deeper/fetches_and_feeddictionary', sess.graph)
writer.close()

sess.close()