
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


def _parse_function(f):
        img = preprocessing.image.load_img(data_dir + "/" + str(f).zfill(6) + ".jpg", color_mode='rgb')
        img = preprocessing.image.img_to_array(img)

        return img
        


def loadimages(ts, vs):
    images = np.array([*range(1, 202600)])
    np.random.shuffle(images)
    trainingimages = images[0:ts]
    testimages = images[ts:ts+vs]
    
    trainingimages = np.array(list(map(_parse_function, trainingimages)))
    testimages = np.array(list(map(_parse_function, testimages)))
    
    trainingimages = trainingimages.astype(np.float32) / 255.
    testimages = testimages.astype(np.float32) / 255.
    
    return trainingimages, testimages

ch = 3
latent_dim = 512
img_height = 218
img_width = 178

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

x_train, x_test = loadimages(20, 20)

x_test = x_test[0:10]

encoded = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded).numpy()

h = open('./saved/saved.txt', 'w')

for i in range(512):
    h.write(str(encoded[0][i]) + "\n")
h.close()


print("Finished saving")