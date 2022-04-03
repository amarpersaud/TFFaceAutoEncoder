# TFFaceAutoEncoder
An autoencoder for faces using TensorFlow, trained on the CelebA dataset

autoencodertest.py trains the model and saves it to a file.

randomface.py generates random faces 

getoutputs.py saves the raw output of the encoder to a file for inspection and modification

showface.py loads the saved raw output and uses the decoder to generate the corresponding face. This allows modification of the values before decoding.