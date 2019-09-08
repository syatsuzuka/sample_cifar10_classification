# sample_cifar10_classification

This is a sample code for cifar10 classification.

## HowToUse

Step.1) Install moduels (cuda, numpy, keras, etc)

Step2.) Download the scripts

Step.3) Execute the training process

 $ python train.py

Step.4) Run the inference process with sample image
You can obtain sample images from the following website (https://www.cs.toronto.edu/~kriz/cifar.html)

 $ python predict.py

 ## Hinto

 You can launch tensorboard by the following command:

  $ tensorboard --logdir logs

You can take a look at the training process from your http://localhost:6006