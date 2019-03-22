import tensorflow as tf

from PIL import Image

original_image_list = ["./working_with_images/images/resize_images/Fundamental_nature.jpg",
                       "./working_with_images/images/resize_images/Moon.jpg",
                       "./working_with_images/images/resize_images/Stonehenge.jpg",
                       "./working_with_images/images/resize_images/sunflower.jpg"]


# Make a queue of file names including all the images specified
# the string_in_pro takes a string of all the images and creates a queue of the filenames
filename_queue = tf.train.string_input_producer(original_image_list)

# read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # A session object in tf is multi-threaded and we want to use the muti-threading capacity to read in our image files
    # To help us with that we use a coordinator class, all the queue related santization and memory , garbage etc operations withing thread are handled by it.
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = []
    for i in range(len(original_image_list)):
        # Read a whole file from the queue , the first returned value in the tuple is the
        # filename which we are ignoring
        _, image_file = image_reader.read(filename_queue)
        # image is the refrence to an image tensor ( a 3 dimensional matrix)
        image = tf.image.decode_jpeg(image_file)

        # gen a tensor of resized images
        image = tf.image.resize_images(image, [224, 224])

        image.set_shape((224, 224, 3))

        # Get an image tensor and print its value

        image_matrix = sess.run(image)

        print(image_matrix.shape)

        # using pillow library to display the resized image
        Image.fromarray(image_matrix.astype('uint8'), 'RGB').show()

        # now we want to convert the 3-D image to a 4-D image where the ist dimension will be the image number
        # so if there is a series of images , the first dimension reperesents the image that we are referencing
        # its like putting the index of an image within the dimensions
        # This is used when there are seperate images , having an individual image tensor is clunky, a single tensor
        # representing multiple images is useful for batch oprations

        # the expand_dims adds new dimention
        image_list.append(tf.expand_dims(image_matrix, 0))

    # Finish off the filename queue coordinator
    coord.request_stop()
    coord.join(threads)

    index = 0
    # write image summary
    summary_writer = tf.summary.FileWriter(
        './working_with_images/resize_images_tensorboard', graph=sess.graph)

    for image_tensor in image_list:
        summary_str = sess.run(tf.summary.image(
            "image-" + str(index), image_tensor))
        summary_writer.add_summary(summary_str)
        index += 1

    summary_writer.close()
