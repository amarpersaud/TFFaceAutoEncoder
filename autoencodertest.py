
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


newnetworkformat = False

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
        '''
        img = tf.keras.preprocessing.image.random_shift(
            img, 0.1, 0.1)
        img = tf.keras.preprocessing.image.random_rotation(
                img, 25)
        '''

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


'''  
#(x_train, x_train_labels), (x_test, x_test_labels) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


x_train =  tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)


x_train = tf.data.Dataset.from_tensors((x_train, x_train))
x_train = x_train.prefetch(60000)

x_test = tf.data.Dataset.from_tensors((x_test, x_test))
x_test = x_test.prefetch(10000)
'''

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

if(not newnetworkformat):
    autoencoder.load_weights(checkpoint_path)

showtestimages = False
trainingsubsets = 1500
epochs = 2
trainingsize = 4096
validationsize = 256
imagecount = trainingsize + validationsize

save_every_n_subsets = 20
savemodel_every_n_subsets = 100

history = []

for j in range(0, trainingsubsets):
    x_train, x_test = loadimages(trainingsize, validationsize)
    if((j % save_every_n_subsets) == 0):
        x_toTest = x_test[0:10]
        encoded = autoencoder.encoder(x_toTest).numpy()
        decoded_imgs = autoencoder.decoder(encoded).numpy()
        decoded_imgs = decoded_imgs.astype(np.float32) * 255
        decoded_imgs = decoded_imgs.astype(np.uint8)
        n=10
        
        rencoded = []
        for i in range(0, 10):
            xA = []
            for k in range(0, latent_dim):
                xA.append(random.uniform(0, 1.0))
            xA = np.array(xA)
            rencoded.append(xA.astype(np.float32))
        encarr = np.array(rencoded)
        rdecoded_imgs = autoencoder.decoder(encarr).numpy()
        rdecoded_imgs = rdecoded_imgs.astype(np.float32) * 255
        rdecoded_imgs = rdecoded_imgs.astype(np.uint8)
        
        plt.figure(figsize=(20, 5))
        for i in range(n):
           # display original
           ax = plt.subplot(3, n, i + 1)
           plt.imshow(x_toTest[i])
           plt.title("original")
           plt.gray()
           ax.get_xaxis().set_visible(False)
           ax.get_yaxis().set_visible(False)
           
           # display reconstruction
           ax = plt.subplot(3, n, i + 1 + n)
           plt.imshow(decoded_imgs[i])
           plt.title("reconstructed")
           plt.gray()
           ax.get_xaxis().set_visible(False)
           ax.get_yaxis().set_visible(False)
           
           # display reconstruction
           ax = plt.subplot(3, n, i + 1 + 2*n)
           plt.imshow(rdecoded_imgs[i])
           plt.title("rand")
           plt.gray()
           ax.get_xaxis().set_visible(False)
           ax.get_yaxis().set_visible(False)
        plt.savefig("subset{}.jpg".format(j))
        plt.close()
    
    x_train = tf.data.Dataset.from_tensor_slices((x_train,x_train)).batch(batch_size)
    x_test = tf.data.Dataset.from_tensor_slices((x_test,x_test)).batch(batch_size)

    
    history = autoencoder.fit(x_train.repeat(),
                    epochs=epochs,
                    steps_per_epoch= (trainingsize // (batch_size * epochs)),
                    validation_steps= (validationsize // (batch_size * epochs)),
                    shuffle=True,
                    validation_data=x_test)
    
    print("Subset {}".format(j))
    dropout = random.uniform(0, 0.15)
    
    if((j % save_every_n_subsets) == 0):
        #save network:
        print("Manually saving weights")
        autoencoder.save_weights(checkpoint_path)
    '''
    if((j % savemodel_every_n_subsets) == 0):
        #save network:
        print("Manually saving model")
        tf.keras.models.save_model(autoencoder, model_path, save_format="tf")
    '''
    
print("fin")
x_train, x_test = loadimages(trainingsize, validationsize)

x_test = x_test[0:10]

if(showtestimages):
    print("Showing original from xt")
    plt.imshow(x_test[0])
    plt.show()

encoded = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded).numpy()

if(showtestimages):
    print("Showing decoded output of network")
    plt.imshow(decoded_imgs[0])
    plt.show()

decoded_imgs = decoded_imgs.astype(np.float32) * 255

if(showtestimages):
    print("Showing *255")
    plt.imshow(decoded_imgs[0])
    plt.show()

decoded_imgs = decoded_imgs.astype(np.uint8)

if(showtestimages):
    print("Showing cast to uint8")
    plt.imshow(decoded_imgs[0])
    plt.show()

n=10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
plt.close()