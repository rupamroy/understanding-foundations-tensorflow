import numpy as np
import tensorflow as tf

# import MNIST Data

from tensorflow.examples.tutorials.mnist import input_data

# store the MNIST data in /tmp/data
# one_hot defines how we want the labels asscoaited with every image to be represented
# this will have the labels as a 10 element vector with all zeros and 1 corresponding to the index of the digit which the image represents
mnist = input_data.read_data_sets("mnist_data/", one_hot=True) # The mnist variable here will hold the input , validation and test datasets

training_digits, training_labels = mnist.train.next_batch(5000) # total has 60,000 images as training data, but we choose 500 so that code runs fine

test_digits, test_labels = mnist.test.next_batch(200) # test of 200 images

 # 784 is the total number of pixels (28 x 28 ) and all are greyscale images 
 # none is the ndex of the image inm a list of images
training_digits_pl = tf.placeholder('float', [None, 784])

test_digit_pl = tf.placeholder('float', [784])

# Step 2 - Nearest calculate using L1 distance
# for explananion see onenote
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl))) 
distance = tf.reduce_sum(l1_distance, axis=1)


# Prediction: get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

# Step -3 Run the algorithm for our test digits and measure accuracy

accurancy = 0.

#Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_digits)):
        nn_index = sess.run(pred, feed_dict={
            training_digits_pl: training_digits,
            test_digit_pl: test_digits[i, :]
        })
        # Get nearest neighbor class label and compare to its true label
        # argmax is required to convert the vector in one-hot notation to a single number [0 0 0 1 0 0 0 0 0 0] => 4
        print("Test:", i, "Prediction:", np.argmax(training_labels[nn_index]),"True Label:", np.argmax(test_labels[i]))

        # Calculate accuracy 
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accurancy += 1./len(test_digits)

    print('Done')
    print('Accuracy:', accurancy)



