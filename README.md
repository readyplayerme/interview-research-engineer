# Vanilla VAE with classifier guidance

## Requirements
This code should run well on any hardware with pytorch and torchvision installed.
```
pip install -r requirements.txt
```

## **Task description**

The task consists of 2 parts:

1. Find bugs in a training code for a simple classifier: [train_classifier.py](./train_classifier.py);
2. Write code for optimizing VAE output with classifier guidance.
    * [latent_optimization_vanilla.py](./latent_optimization_vanilla.py)
    * This means that VAE should produce the number that we are asking for. You should use pre-trained classifier for that in this setup.