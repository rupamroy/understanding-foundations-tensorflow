import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plot
import os

filename = "./working_with_images/DandelionFlower.jpg"

image = mp_img.imread(filename)

# image is a numpy array, 

print("image shape: ", image.shape)
print("Image array: ", image)

plot.imshow(image)
plot.show()
# Image is now a 3 dimentional matrix
x = tf.Variable(image, name="x")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # tf.transpose can be used to transpose any n dimentional matrix
    # It can flip arround any axis of any dimensional matrix
    # Original Axis indexes for a given image are 0,1,2
    # With [1, 0, 2] the first and the second axis are swiped 
    # this swaps the width and the height but leaves the third axis (which is the pixel values)

    # transpose = tf.transpose(x, perm=[1, 0, 2])

    # above transpose works for any metrix
    # for images we have special , this is simple
    transpose = tf.image.transpose_image(x)

    result = sess.run(transpose)

    print("Transposed image shape: ", result.shape)
    plot.imshow(result)
    plot.show()


    


