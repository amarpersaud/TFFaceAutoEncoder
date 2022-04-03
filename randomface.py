
import tensorflow.compat.v1 as tf 
tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 
#import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import pathlib 
import os 
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing
import pathlib

data_dir = "./img_align_celeba"
checkpoint_path = "./cp.ckpt"
model_path = "./model"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

ch = 3
#latent_dim = 512
latent_dim = 512
img_height = 218
img_width = 178
batch_size = 32

dropout = 0.12


class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Conv2D(4, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(8, (3, 3), activation='relu'),    
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dropout(dropout),
      layers.Dense(latent_dim*2, activation='relu'),
      layers.Dense(latent_dim*2, activation='elu'),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(latent_dim, activation='elu'),
      layers.Dense(latent_dim*2, activation='sigmoid'),
      layers.Dense(img_height * img_width*ch, activation='sigmoid'),
      layers.Reshape((img_height, img_width, ch))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.load_weights(checkpoint_path)

encoded = []
for i in range(0, 20):
    xA = []
    for j in range(0, latent_dim):
        xA.append(random.uniform(0, 1.0))
    xA = np.array(xA)
    encoded.append(xA.astype(np.float32))
encarr = np.array(encoded)
decoded_imgs = autoencoder.decoder(encarr).numpy()
decoded_imgs = decoded_imgs.astype(np.float32) * 255
decoded_imgs = decoded_imgs.astype(np.uint8)

print("Decoded. Displaying:")

n=10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(decoded_imgs[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i + n])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.savefig("generatedfaces.jpg")