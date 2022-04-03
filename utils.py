from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import tensorflow as tf
import numpy as np
from tensorflow import io
import variables
import models
from tkinter import filedialog # Somehow, import tkinter as tk didn't work
from model.srgan import generator
from model.common import resolve_single
from PIL import Image

def get_img(img_path):
    img = io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = variables.max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    return img, new_shape


def get_content_img():
    img_path = filedialog.askopenfilename(initialdir="/", title="Choose content image",
                                            filetypes=(("all files", "*.*"), ("png files", "*.png")
                                                       ,("jpg files", "*.jpg")))
    img, new_shape = get_img(img_path)
    variables.content_image = tf.image.resize(img, new_shape, antialias=True)
    variables.content_image = variables.content_image[tf.newaxis, :]


def get_style_img():
    img_path = filedialog.askopenfilename(initialdir="/", title="Choose style image",
                                             filetypes=(("all files", "*.*"), ("png files", "*.png")
                                                        ,("jphg files", "*.jpg")))

    img, new_shape = get_img(img_path)
    variables.style_image = tf.image.resize(img, new_shape, antialias=True)
    variables.style_image = variables.style_image[tf.newaxis, :]


def get_output_dir():
    variables.output_dir = filedialog.asksaveasfilename(initialdir="/", title="Select output location")


def upscale_gan(img):
    model = generator()
    model.load_weights('weights/weights/srgan/gan_generator.h5')
    variables.textVar.set("Using GAN to improve quality")
    variables.status_lbl.update()
    sr = resolve_single(model, img)
    return sr

def commence():
    variables.textVar.set("Commencing")
    variables.status_lbl.update()

    # Updating parameters
    variables.height = int(variables.height_sel.get())
    variables.width = int(variables.width_sel.get())
    variables.steps_per_epoch = int(variables.steps_epoch_sel.get())
    variables.epochs = int(variables.epoch_sel.get())
    variables.style_weight = float(variables.style_weight_sel.get())
    variables.content_weight = float(variables.content_weight_sel.get())
    variables.tv_weight = float(variables.tv_weight_sel.get())

    variables.textVar.set("Variables set")
    variables.status_lbl.update()
    mdl = models.StyleContent(variables.content_image, variables.style_image)
    variables.textVar.set("Model created")
    variables.status_lbl.update()
    mdl.train()


def save_img(image_batch, epoch):
    file_name = f'{variables.output_dir}{epoch}.jpg'

    for i in image_batch:
        i = tf.image.resize(i, [variables.height,variables.width])
        img = tf.image.encode_jpeg(tf.cast(i * 255, tf.uint8), format='rgb')
        tf.io.write_file(file_name, img)
        img = upscale_gan(load_image(file_name))
        img = tf.keras.preprocessing.image.array_to_img(img)
        img.save(file_name)

def load_image(path):
    return np.array(Image.open(path))
