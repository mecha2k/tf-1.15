# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
import sys
import tensorflow as tf
import os

print(tf.__version__)
print(os.getcwd())
print(sys.path)

physical_gpus = tf.config.experimental.list_physical_devices("GPU")
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(physical_gpus), "Physical GPUs, ", len(logical_gpus))
    except RuntimeError as e:
        print(e)


"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
import config


tflib.init_tf()

# Load pre-trained network.
url = "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ"  # karras2019stylegan-ffhq-1024x1024.pkl
# url = "https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf"  # CelebA-HQ dataset at 1024×1024
# url = "https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF"  # LSUN Bedroom dataset at 256×256
# url = "https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3"  # LSUN Car dataset at 512×384


with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

# Print network details.
Gs.print_layers()

n_image = 20
os.makedirs(config.result_dir, exist_ok=True)

for i in range(n_image):
    # Pick latent vector.
    rnd = np.random.RandomState()
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    png_filename = os.path.join(config.result_dir, f"result{i+1:0>2d}.png")
    PIL.Image.fromarray(images[0], "RGB").save(png_filename)

print(n_image, "images are generated.")
