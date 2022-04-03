from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import variables
import tensorflow as tf
from tensorflow import linalg
from tensorflow.keras import applications
from tensorflow.keras import models
import h5py


def get_vgg_layers(layer_names):
    vgg = applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = models.Model([vgg.input], outputs)
    return model


def calculate_gram_matrix(tensor):
    input_shape = tf.shape(tensor)
    result = linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def compute_loss(outputs, targets):
    return tf.add_n([
        tf.reduce_mean(tf.square(outputs[name] - targets[name]))
        for name in outputs.keys()
    ])


def get_high_frequencies(img):
    x = img[:, :, 1:, :] - img[:, :, :-1, :]
    y = img[:, 1:, :, :] - img[:, :-1, :, :]
    return x, y


def variation_loss(img):
    x, y = get_high_frequencies(img)
    return tf.reduce_mean(tf.square(x)) + tf.reduce_mean(tf.square(y))


def get_style_content_loss(outputs, content_targets, style_targets, content_layers,
                           style_layers, img):
    content_loss = compute_loss(outputs['content'], content_targets)
    style_loss = compute_loss(outputs['style'], style_targets)

    content_loss *= variables.content_weight / len(content_layers)
    style_loss *= variables.style_weight / len(style_layers)

    total_loss = style_loss + content_loss
    total_loss += variables.tv_weight * variation_loss(img)

    return total_loss
