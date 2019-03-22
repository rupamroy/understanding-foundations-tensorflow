import tensorflow as tf

# y = Wx + b

W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')

y = W * x + b

# Initiallize all variables defined before using them
init = tf.global_variables_initializer() # will all variables

with tf.Session() as sess:
    sess.run(init)

    print("Final result: Wx + b =", sess.run(y, feed_dict={x:[10, 100]}))

# Intiallize only those variable which you may need
s = W * x

# single variable initiallization
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)

    # print("This will not work: Wx + b =", sess.run(y, feed_dict={x:[10, 100]}))
    print("This will work: s=", sess.run(s, feed_dict={x:[10, 100]}))

number = tf.Variable(2)
mutiplier = tf.Variable(1)

init = tf.global_variables_initializer()

# variable number is being reassigned a value
# here result does not hold the new value of the number, its just a reference to the calculation node to be used to run the assignment in a session
result = number.assign(tf.multiply(number, mutiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("Result number * multiplier = ", sess.run(result))
        # assign_add adds a number to the given variable and assigns the result to the variable
        print('increment multiplier, new value = ', sess.run(mutiplier.assign_add(1)))


