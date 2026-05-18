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