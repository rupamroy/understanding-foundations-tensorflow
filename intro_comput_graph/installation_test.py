# import tensorflow
import tensorflow as tf

sess = tf.Session() 

# verify we can print a string
hello = tf.constant("Hello Pluralsight from Tensorflow")

print(sess.run(hello))


# verify simple maths

a = tf.constant(20)
b = tf.constant(22)

print('a + b = {0}'.format(sess.run(a + b)))