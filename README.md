# CoCa Capsule Network based Contrastive Learning of Unsupervised Visual Representations

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg) 
[![harshpanwar](https://img.shields.io/twitter/follow/harsh__panwar?style=social)](https://mobile.twitter.com/harsh__panwar)

Official PyTorch Implementation of my MSc Thesis - [Capsule Network based Contrastive Learning of Unsupervised Visual Representations](https://arxiv.org/pdf/2209.11276.pdf).

Usage:

1. Download the required modules using the command 'pip install -r requirements.txt'.
2. Run the python file main.py to start training.

```
optional arguments:
--temperature                 Temperature used in softmax [default value is 0.2]
--k                           Top k most similar images used to predict the label [default value is 50]
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
--num_caps                    Number of capsules per layer [default value is 32]
--caps_size                   Number of neurons per capsule [default value is 64]
--depth                       Depth of additional layers [default value is 1]
--planes					  Starting layer width [default value is 16]
```

3. Once the training is completed the trained weights and the training loss, top-1 and top-5 accuracy will be stored in a folder called 'results'.

