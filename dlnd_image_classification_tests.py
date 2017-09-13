from unittest import TestCase

import dlnd_image_classification as ic
import problem_unittests as tests


class Project2Tests(TestCase):
    @staticmethod
    def test_should_normalize():
        tests.test_normalize(ic.normalize)

    @staticmethod
    def test_should_hot_encode():
        tests.test_one_hot_encode(ic.one_hot_encode)

    @staticmethod
    def test_should_build_network():
        import tensorflow as tf

        tf.reset_default_graph()
        tests.test_nn_image_inputs(ic.neural_net_image_input)
        tests.test_nn_label_inputs(ic.neural_net_label_input)
        tests.test_nn_keep_prob_inputs(ic.neural_net_keep_prob_input)

    @staticmethod
    def test_should_create_conv2d_maxpool():
        tests.test_con_pool(ic.conv2d_maxpool)

    @staticmethod
    def test_should_create_flatten():
        tests.test_flatten(ic.flatten)

    @staticmethod
    def test_should_create_fully_conn():
        tests.test_fully_conn(ic.fully_conn)

    @staticmethod
    def test_should_create_conv_net():
        tests.test_conv_net(ic.conv_net)
