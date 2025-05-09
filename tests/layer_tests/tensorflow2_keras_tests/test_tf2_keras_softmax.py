# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasSoftmax(CommonTF2LayerTest):
    def create_keras_softmax_net(self, input_names, input_shapes, input_type, ir_version):
        """
               Tensorflow2 Keras net:                     IR net:
                      Input               =>               Input
                        |                                    |
                     Softmax                               Softmax
        """
        # create TensorFlow 2 model with Keras Softmax operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type,
                            name=input_names[0])  # Variable-length sequence of ints
        y = tf.keras.layers.Softmax()(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # create reference IR net
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_precommit = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_softmax_float32(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_softmax_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)

    test_data_float32 = [dict(input_names=["x1"], input_shapes=[[5, 4]],
                              input_type=tf.float32),
                         dict(input_names=["x1"], input_shapes=[[5, 4, 8]],
                              input_type=tf.float32),
                         dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
                              input_type=tf.float32),
                         dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]],
                              input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_softmax_float32(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_softmax_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
