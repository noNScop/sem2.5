I devided the task of training a classification model on food101 into 3 notebooks. In addition there are also:
- tensorboard_stats - I saved some results of experiments in tensorboard format
- engine.py - script containing my functions that I was using frequently
- data - a place for food101 data to be download, 3 images for classification are already there (I used them in the inference at the end)


### 1) EffNet_B5 vs EffNetV2-S on Food101 images
Here I prepared a 20% subset of full data to speed up experimentation, then I implemented learning rate finder that works simillar to learning rate finder in fastai. I also created a training loop and finally compared the performance of EffNet_B5 and EffNetV2-S. I decided to use EffNetV2-S on the full dataset.
### 2) Hyperparameters tuning for EffNetV2-S on Food101
I continue on using 20% subset of data and the same LRFinder class and train function I implemented in part 1. I made an attempt to optimize the following hyperparameters:
- optimizer and its learning rate,
- data augmentation,
- batch_size,
- weight decay.
It later turned out that the weight decay I found well, didn't generalise to the final training on full dataset, but the rest worked very well.
### 3) The final training on the full food101 dataset
Actually a lot more than that! First it turned out that I missed one important detail in all my previous considerations - batch_norm layers. It appears that it is better to put them (and the rest of the model) in .eval() mode, so they can use running statistics accumulated during pre-training on ImageNet. I actually have done the final training 3 times to test different lr schedules and weight decays, I also repeated 2 final experiments from previous notebooks to see the impact of putting batch_norm layers in .eval() mode there. I conclude that tuning weight decay on subset of data or on a fraction of final epochs makes very little sense, I guess it is just better to go with some intuitively good value and see how it works out. In the end I have also made some predicitons with my trained model to see it in action.
