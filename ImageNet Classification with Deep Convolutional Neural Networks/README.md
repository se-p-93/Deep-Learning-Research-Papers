## ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, NIPS, 2012

Here I attempt to duplicate the results in [this paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). The AlexNet architecture was developed for the Imagenet ILSVRC-2012 competition had a top-5 error rate of 15.3%, compared to 26.2% achieved by the second-best entry. It's one of main factors that led to the deep learning research to skyrocket since 2012, and introduces the common deep convolutional neural network architecture used nowadays.

I implemented the architecture with Keras. First, I go over a quick overview
of the paper.

## Paper overview
### Architecture
- 8 layers. 5 convolutional + pooling and 3 fully connected layers.
- The original network trained on 2 parallel GTX 580 GPUs (3 GB memory).
- Used Stochastic Gradient Descent with momentum, weight decay and learning rate annealing.

### Dataset
- 1.2 million images taken from ImageNet. 1 million train, 150.000 test and 50.000
validation of various sizes and resolutions.
- Resize smaller dimension to 256 pixels, then cropped out the central 256x256 from the resulting image. 
- Substracted the mean image from all the images in the training set.
- Takes a `227 x 227 x 3` image as input. The paper says 224x224 but it's a mistake, it makes sense with a 227x227, having a kernel of 11x11 and a s=4, then (227-11)/4+1=55.
- 1000 classes.

### Activation Unit
Uses Rectified Linear Unit (ReLu) as activation unit, instead of a more traditional
`tanh` unit, which takes a long time to saturate. This speeds up the training process
several times, thus, allowing the use of a much bigger training set.

### Pooling layers
They use an overlapping pooling technique. If `s` is the number of units the pooling
grip are spaced apart and `z` is the pooling neighborhood `z x z`, then if `s < z`
it's proven to be harder to overfit.

### Dropout
It randomly drops a percentage of the neurons (50%, in this case) in the fully
connected layers in each iteration. It was the first time dropout had been used
in a practical model.

### Data Augmentation
Takes random images and performs random techniques to alter the images. This
can be reflection, translation or patch extraction. It's used to prevent overfitting.

### Local Response Normalization
The activation output of each neuron is divided by a term proportional to the
sum of squares of activations of the neurons in some neighboring channels. This
helps the generalization of the results.

## Implementation 

- The original architecture was trained in parallel with two GPUs but I implemented it for a single GPU.
- Batch normalization was used instead of local response normalization.
- No data augmentation was performed.
- The network was trained on a GTX 1060 (6 GB memory).
- Dataset used was the ILSVRC-2010 from ImageNet.
- First I tested the network on the CIFAR-10 dataset.

## Results

### CIFAR-10

The implementation for this is found on the cifar-Alexnet notebook. The accurary on the validation set was 0.809.

![](results/cifar-10.png)

### ILSVRC-2012

In progress.