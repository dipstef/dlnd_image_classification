import numpy as np
import tensorflow as tf


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return (x - np.min(x)) / np.ptp(x)


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels

    >>> np.array([[2,3,4,0], [4,5,6,7]]).reshape(-1, 1)
        array([[2],
               [3],
               [4],
               [0],
               [4],
               [5],
               [6],
               [7]])
    """
    # return np.eye(10)[np.array(x).reshape(-1)]
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(n_values=10)

    elements = np.array(x).reshape(-1, 1)

    return enc.fit_transform(elements).toarray()


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, shape=[None, *image_shape], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=[None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool

    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    _, _, _, input_depth = x_tensor.get_shape()
    weights = tf.Variable(tf.truncated_normal(shape=(*conv_ksize, int(input_depth), conv_num_outputs),
                                              stddev=0.1))

    bias = tf.Variable(tf.zeros(conv_num_outputs))

    conv = tf.nn.conv2d(x_tensor, weights, strides=[1, *conv_strides, 1], padding="SAME")
    conv = tf.add(conv, bias)
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv, ksize=[1, *pool_ksize, 1], strides=[1, *pool_strides, 1], padding="SAME")

    return conv


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    return tf.contrib.layers.flatten(x_tensor)


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=tf.nn.relu)


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=None)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    #    Play around with different number of outputs, kernel size and stride
    x = conv2d_maxpool(x, conv_num_outputs=64, conv_ksize=(3, 3), conv_strides=(1, 1), pool_ksize=(2, 2), pool_strides=(2, 2))

    x = tf.nn.dropout(x, keep_prob)

    x = conv2d_maxpool(x, conv_num_outputs=128, conv_ksize=(3, 3), conv_strides=(1, 1), pool_ksize=(2, 2), pool_strides=(2, 2))
    x = conv2d_maxpool(x, conv_num_outputs=256, conv_ksize=(3, 3), conv_strides=(1, 1), pool_ksize=(2, 2), pool_strides=(2, 2))

    x = flatten(x)

    #    Play around with different number of outputs
    x = fully_conn(x, 512)
    x = tf.nn.dropout(x, keep_prob)
    x = fully_conn(x, 256)
    x = tf.nn.dropout(x, keep_prob)

    #    Set this to the number of classes
    out = output(x, 10)

    return out
