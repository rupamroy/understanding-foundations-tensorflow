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

        # Some operations that can be done on images
        image = tf.image.flip_up_down(image)

        image = tf.image.central_crop(image, 0.5)

        # Get an image tensor and print its value
        image_array = sess.run(image)
        print(image_array.shape)


        # converts a numpy array of the kind (224, 224, 3) to a Tensor of shape (224, 224,3)
        image_tensor = tf.stack(image_array)

        print(image_tensor)
        image_list.append(image_tensor)

       
    # Finish off the filename queue coordinator
    coord.request_stop()
    coord.join(threads)

    # tf.stack converts a list of rank-R tensors into one rank-(R+1) tensor
    images_tensors= tf.stack(image_list)
    print(images_tensors)

    # This is the output
    # (224, 224, 3)
    # Tensor("stack:0", shape=(224, 224, 3), dtype=float32)
    # (224, 224, 3)
    # Tensor("stack_1:0", shape=(224, 224, 3), dtype=float32)
    # (224, 224, 3)
    # Tensor("stack_2:0", shape=(224, 224, 3), dtype=float32)
    # (224, 224, 3)
    # Tensor("stack_3:0", shape=(224, 224, 3), dtype=float32)
    # Tensor("stack_4:0", shape=(4, 224, 224, 3), dtype=float32)

    summary_writer = tf.summary.FileWriter(
        './working_with_images/image_transformations_tensorboard', graph=sess.graph)

    # write all images in one go
    summary_str = sess.run(tf.summary.image('images', images_tensors, max_outputs=4)) # without max_outputs only default 3 images will show up in the tensorboard
    summary_writer.add_summary(summary_str);

    summary_writer.close()
