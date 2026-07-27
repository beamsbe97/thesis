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


# 22/4
understand why randomness was used in eval
- ran eval on verbs in quick succession, (4 runs within 30min) = no randomness, all results are the same

call the eval function in train.py (cause its not getting called currently)

# 29/4
- find out what features are used for action localization
newer papers use VideoMAE
HACS: VideoMAE features
FineAction: VideoMAE features
EpicKitchen: SlowFast features (adatad, )

- look up adatad(using video MAE), find papers that cite adatad and actionformer and see which one people are using
Tridet used in SOTA comparisons (322 citations, IEEE '23) uses SlowFast features for EpicKitchen
Actionformer is still the baseline to beat in SOTA comparisons
More recent papers use Actionformer as detection head
Below papers use VideoMAE to extract features and ActionFormer as detection head
    - E2E-TAD mentioned quite a lot (144 citations, IEEE)
    - Temporal Action Detection Model Compression by Progressive Block Drop (published CVF march '25, 9 citations)

- swap slowfast features for v-jepa


# 8/5
- 
- extract features from v-jepa2 specifically, cause adatad++ has 
- see how the frames are sampled by v-jepa2 
- aim for 1 feat vector per half a second (same as slowfast)
64 frame window, stride of 30frames. clips are 60fps

# 10/6
- try to extract with 30fps
- if have features, train with actionformer
- try to understand

# 24/6
- for 50fps videos, reencode with ffmpeg and then   extract features again
- read up on how neco implements augmentation, think about how we can adapt it to ours

# 8/7
- think through why which augmentation works, what it is doing
- 2 views are teaching that they are the same thing. so think of how augmentation can be done temporally
- check on the params of the augmentations
    size_crops: size of global and local crop
    nmb_crops: number of global and local crop
    min_scale_crops: the lower bound for the random area of the global and local crops before resizing
    max_scale_crops: the upper bound for the random area of the global and local crops before resizing
    jitter_strength: the strength of jittering for brightness, contrast, saturation and hue
    min_intersection: minimum percentage of intersection of image ares for two sampled crops from the
        same picture should have. This makes sure that we can always calculate a loss for each pair of
        global and local crops.
    blur_strength: the maximum standard deviation of the Gaussian kernel

    
- self-sup : making up learning signal, hard enough to learn, easy enough that it can learn
- think about whether augment on pixel space or features


# 15/7
- UPDATES: 
    reencoded videos have about the same expected number of vectors (5% difference on avg)
- TO DO:
    visualise training losses
    see if anything wrong with curves
    check what input size into NeCo is
    try to understand what augmentation is teaching the model to do