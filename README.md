# TFFaceAutoEncoder
An autoencoder for faces using TensorFlow, trained on the CelebA dataset

Code written for running on AMD GCN cards using TensorFlow 1

## Files

autoencodertest.py trains the model and saves it to a file.

randomface.py generates random faces 

getoutputs.py saves the raw output of the encoder to a file for inspection and modification

showface.py loads the saved raw output and uses the decoder to generate the corresponding face. This allows modification of the values before decoding.

## Training Output

Below shows several examples of the output of the program while training

First the output is gray due to the network being initially randomized or zeroed.

![Initial output](https://github.com/amarpersaud/TFFaceAutoEncoder/blob/main/Examples/subset0.jpg)

After substantial training, the network begins to show an output that somewhat matches the input

![Substantial training](https://github.com/amarpersaud/TFFaceAutoEncoder/blob/main/Examples/subset1400.jpg)

The training ouput begins to look like a blurred version of the input values. With more training and a better network structure, a higher quality output could be achieved. 

The third row which shows faces generated using random values does not show viable human faces. This is likely due to serveral factors: limited training time, values outside the range that the network uses, and the modification of values in the encoded space that the network either does not use, or values which it does not use simultaneously, and thus results in spurious inputs that interrupt the network, or does not modify them within the full range of possible inputs (e.g. [0,1] instead of [0,10] would result in a limited input range that does not properly reflect the encoder's output, resulting in many similar looking faces).

## Modified Output

The output of a single encoded face that is then modified several times before decoding is shown below in several states

Initially the face is unmodified and looks similar to the output of the training

![Unmodified](https://github.com/amarpersaud/TFFaceAutoEncoder/blob/main/Examples/f5.jpg)

With slight modification, the output begins to change. The mouth is now closed and the teeth are not shown

![Slight modification](https://github.com/amarpersaud/TFFaceAutoEncoder/blob/main/Examples/f9.jpg)

With substantial modification to the input of the decoder, the hair and eyes change substantially, as well as the face direction.

![Substantial modification](https://github.com/amarpersaud/TFFaceAutoEncoder/blob/main/Examples/f10.jpg)

Modifying the input manually doesn't produce a realistic result due to multiple values in the encoded space changing values that a human would perceive as identical, such as hair color on the left and right halves of the image as shown above. Constricting the number of dimensions in the encoded space, a better suited network structure and more training could significantly improve the results.
