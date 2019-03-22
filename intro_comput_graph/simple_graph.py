import tensorflow as tf

a = tf.constant(6, name='constant_a')
b = tf.constant(3, name='constant_b')
c = tf.constant(10, name='constant_c')
d = tf.constant(5, name='constant_d')

# a , b, c and d are tensors or edges in our graph
# constants in tensorflow is immutable
# The name helps us identidfy these contants when we see them in tensorboard

# mul, div and addn are computational nodes
mul = tf.multiply(a, b, name='mul')

div = tf.div(c, d, name='div')

addn = tf.add_n([mul, div], name='addn')

print(addn) #Priting a tensor variable just gives the name the shape and datatype and not the value it is holding. 

sess = tf.Session()

print(sess.run(addn))

writer = tf.summary.FileWriter('./m2_example1', sess.graph)
# To see this in tensorboard on the python enn command line type - tensorboard --logdir="m2_example1"
writer.close()

sess.close()