import tensorflow as tf

sess = tf.Session()

zeroD = tf.constant(5)
print(sess.run(tf.rank(zeroD)))

oneD = tf.constant(["How", 'are', 'you'])
print(sess.run(tf.rank(oneD)))

twoD = [[1.0,2.0,3.0],[1.5,2.9,3.1]]
print(sess.run(tf.rank(twoD)))

threeD=tf.constant([[[1,2],[3,4],[5,6],[7,8]]])

print(sess.run(tf.rank(threeD)))

sess.close()
