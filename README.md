# Distributed Training Lab (TensorFlow)

This project demonstrates training a neural network using TensorFlow and explores how distributed training works.

## What I did
- Trained a CNN model on the MNIST dataset using TensorFlow and Keras
- Set up a simulated multi-worker configuration using `TF_CONFIG`
- Ran the model in both single-worker and attempted multi-worker modes
- Added custom performance tracking to the training script

## My changes
I modified the training script to make the lab more meaningful by adding performance analysis. Specifically, I:

- Measured total training time using a timer
- Printed the final training accuracy after the model finished
- Structured the output to clearly show a training summary

This turns the lab from just running a model into something that actually evaluates performance.

## Results
- Final accuracy: ~0.60  
- Training time: ~8 seconds  

## Notes
I set up the multi-worker configuration and launched multiple workers locally. However, full distributed execution was unstable in the Colab environment, so I focused on completing the workflow and analyzing training performance instead.
