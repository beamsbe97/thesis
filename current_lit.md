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
    -- Global crops: sampled from 25%–100% of the original image area, then resized to 518×518.
    -- Local crops: sampled from 5%–25% of the original image area, then resized to 98×98.
    -- Color jitter, grayscale, Gaussian blur: remove pixel-level rgb/texture so the network cannot cheat. forcing it to match by semantic, NO appearance shortcut
    -- two different views (different scale, different crop location, different appearance) to force the model to learn something beyond "match identical pixels," but the loss (patch neighbor consistency) requires that the same underlying image region be represented in both views so you have a supervision target

    try to understand what augmentation is teaching the model to do

# 30/7
- find out what the crop overlap is on average (between global and local)
     average overlap is ~14–18% of the full image

- augmentation styles:
    - spatial crops
    - temporal crops
    - frame rates
    - combination of spatial and temporal

- before exp, look at the few temporal crops and see what they look like


- keep in mind the clip size input into the model, since it affects what 25% means
- also batch size 
- get the baseline


# 10/8
- train actionformer NeCo style FIRST and then conduct training AFTER NeCo SSL

- see if the loss curves start lower/same and ends up at the same point

- if theres time, find MAE features or finetuned VJEPA2

# 24/9
- joint fine-tune ablation on pipeline 2 (SSL post-trained encoder + supervised heads, all unfrozen, 6 ep @ lr 2e-5)
    - SSL-post + joint FT: 22.45 final (22.45-22.61 over epochs) -- no recovery vs frozen heads (22.55)
    - control (baseline, same FT, no SSL): 22.97 (22.94-23.02)
    - => head mismatch is NOT the explanation; SSL post-training hurts the encoder itself
- launched V-JEPA2 through pipeline 3 (SSL from random init -> supervised)
- found pre-extracted EPIC-finetuned VideoMAE-L features (OpenTAD), verb + noun sets, 1024-d, stride 8
    - OpenTAD's ActionFormer SlowFast verb = 24.93 avg mAP (> our 23.07)



- consider saving checkpoints at earlier epochs, since downstream task can be similar to pretraining
- next try spatial crops 

# 
- check if if epoch 1 they start that the same loss
- see how much the features change after the post training
- get the thesis cover page
- focus more on why it doesnt work (important)

# 22/10
- encoder-similarity diagnostic (tools/visualize_encoder_similarity.py): linear
  CKA per FPN level, per-position cosine, PCA, weight cosine between trained encoders.
- SSL-init->supervised vs baseline (after supervised): CKA 0.93 (SlowFast) / 0.97
  (VideoMAE); per-position cosine 0.71 / 0.87. Seed control (2 baseline seeds):
  CKA 0.90, cosine ~0. So the SSL pair is about as similar as two seeds.
- weight cosine is dominated by the SHARED INIT SEED, not training: a fresh init
  with the training seed has wcos 0.974 with the trained supervised model (0.55 for
  arbitrary init). Baseline and pipeline-3 shared init_rand_seed -> artifact.
  (corrects earlier over-reading)
- BIG ONE: SSL stage-A barely trains the encoder. rel L2 change from random init:
  supervised backbone 0.238; SSL backbone 0.0046 (SlowFast / VideoMAE spatial /
  VideoMAE temporal all); SSL head 0.12. So SSL trains only the projection head;
  the "SSL encoder" is ~random init -> pipeline 3 = supervised from ~random init =
  baseline. Mechanically explains all the null results.
  causes: encoder lr 2e-5 vs head 2e-4 (head_lr_scale 10); global grad clip 1.0
  starves the encoder; NeCo satisfiable by the head; only 12 epochs.
- next: raise encoder lr (~1e-4), reduce/remove grad clip, lower head_lr_scale,
  longer schedule; check backbone rel_delta >10%; then re-run stage A->B. If the
  encoder now trains and mAP is still flat -> frozen-features claim holds; if it
  improves -> the earlier nulls were an artifact.

# 23/10
- implemented the SSL-training fix and re-ran it (VideoMAE spatial):
  encoder lr 2e-5 -> 1e-4, head_lr_scale 10 -> 1, clip_grad_l2norm 1 -> 5,
  epochs 12 -> 20. Config `ssl_epic_videomae_verb_spatial_v2.yaml`.
- stage A v2 (job 5318111): val loss 2.19, agreement 0.889 (v1: 2.80 / 0.843).
  backbone rel_delta 0.056 (v1 0.0046) = 12x more, but still 4x less than the
  supervised 0.238. So the fix helps, but NeCo still gives weak encoder
  gradients (expressive head / EMA self-distillation is easy).
- stage B v2 launched (job 5318166) -> compare to 29.60 baseline / 29.38 v1.
- if still null: go more aggressive (higher encoder lr, weaker projection head,
  no clip) before concluding anything about the features.
- RESULT stage B v2: **avg mAP 28.65** (best 28.83 @ ep16) -> WORSE than v1
  (29.38) and baseline (29.60). So forcing the encoder to train 12x more with
  NeCo does not help; it slightly hurts. Caveats: 1 seed; encoder still only
  5.6% vs supervised 23.8%, so not a fully-trained-encoder test. This weakens
  the "SSL just wasn't training the encoder" explanation and points at the
  NeCo objective / features instead. Next: either a much more aggressive v3
  (stronger lr / weaker head / no clip) or pivot to feature-level ideas
  (`feature_ideas.md`).


  # 

  - check if SSL loss curve starts lower or higher than normal training (takes more effort just to get to reuglar supervised learning baseline)
  - plot the same graph for NeCo
  - check what head/neck/backbones are actually, and why supervised has 0 head weight change, what is the head? for the specific task? reg/clf head? or what is it?