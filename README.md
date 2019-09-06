## What is it ? 

It's model from a [A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation](https://arxiv.org/pdf/1512.02134.pdf) article. I used it to train DispNet model for solving a stereo disparity problem. 
Using a FlyingThings3D dataset I obtained *endpoint error* about **2.8**. 
During training I used next random augmentations: crop, rotation in range -5 to 5 degrees, color jitter, 
gaussian noise. I did not make any hyperparamters search because of lack of computing resources and time.


## Stereo disparity task

Having two images of a scene from left and right cameras we want to get depth map of the scene.
