ASFormer (2021, 350 citations) - used in SOTA comparisons in multiple temporal action segmentation papers

DiffAct (2023, 165 citations) - used in SOTA comparisons in more recent papers


# 15/4
have a script to put annotations in kitchen subs - done
try to get overlapping actions

check how features are done, how the frames are chosen(consecutive frames or what) - done
Features are extracted from SlowFast pretrained on epickitchens
32 frame window, stride of 16 frames. clips are 30fps
1 feature vector per ~0.53 seconds (16/30)

plot the actionformer training curves to see what normal training with SlowFast looks like - done
