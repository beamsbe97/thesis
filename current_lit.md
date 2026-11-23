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

# 25/9
- seeds (SlowFast, n=3): baseline 23.38 +- 0.75 (22.84-24.23), pipeline 3 23.65 +- 0.08
    - +0.27, not significant -> the +0.49 "beats paper" was baseline seed noise
    - but pipeline 3 is MUCH more stable across seeds (range 0.16 vs 1.39)
- VideoMAE-L (OpenTAD, EPIC-ft): baseline 29.60, post-train 29.24, pretrain 29.38
    - post-training hurts on both tracks now; pretraining no gain
- frame-rate aug (feature-level, per-view rate log-U[0.5,2]) + pipeline 3: 24.45 (1 seed!)
    - +0.8 over pipeline 3 without it; first SSL variant above the baseline distribution
    - next: replicate with seeds, then try on VideoMAE
    - (synced from ALICE 2026-11-23; these SSL numbers are pre-fix, see 22-23/11)



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

# 13/11 — answered the "does the supervised loss start lower after SSL?" question
  - new tool `tools/analyze_loss_start.py`; figure
    `figs/loss_start_comparison.png`; numbers in `report.md` §3.4.
  - seed-matched pairs share head init + data order, so the epoch-1 supervised
    loss isolates the encoder init. Result: SSL init does NOT start lower.
    Δ(SSL−random) at epoch 1: SlowFast +0.12%/+0.16% (seeds 1/2), VideoMAE
    spatial +0.19%, spatial-v2 +0.56%, V-JEPA2 −0.20%. 5/6 start *higher*.
  - V-JEPA2 starts lower but has worse mAP (17.40 vs 17.49) -> a lower start is
    not evidence of transfer (can be train-set overfitting). So the loss-curve
    probe is consistent with the CKA result (§6.3) but is a weak, confounded
    probe on its own.
  - caveat: batch size 2; single minibatch loss varies 1.97/6.41/2.82 across
    seeds for the same recipe — only epoch means are usable.
  - next: for NeCo -> spatial segmentation, see the options discussed
    (initial-loss probe, frozen/linear head, CKA to seg-trained backbone,
    gradient cosine, NeCo-neighbour vs GT-label agreement).

# 15/11 — NeCo in its own setting (DINOv2 ViT-S/14 -> PASCAL VOC segmentation)
  - separate tree on Alice: `/zfsstore/user/s4561341/neco_diag`, official
    `neco_repo` + released checkpoints; VOC 2012 trainaug (10582) / val (1449).
  - protocol: frozen backbone, 1x1 conv on patch tokens -> 100x100, SGD 0.01,
    wd 1e-4, 25 ep, step decay @20, seed 400 for *both* runs (same head init +
    data order).
  - **option 2 (linear probe)**: DINOv2 79.65 best (79.34 final) vs NeCo **81.62**
    (81.24 final) = +1.97 / +1.90 pp. Paper's NeCo ViT-S/14 linear = 81.4, so the
    pipeline reproduces the published number; the DINOv2 control is our own run.
  - **batch-size trap**: the first NeCo probe ran at bs 64 (165 steps/epoch) vs
    DINOv2 at 128 (82 steps). That inflated the epoch-0 gap to −35% train CE and
    also nudged the final mIoU (81.67). Reran NeCo at bs 128 (paper protocol).
    Lesson: only quote epoch-0 losses within a matched step count.
  - **epoch-0 CE (bs 128 both)**: train 0.6103 -> 0.5785 (−5%), val 0.1993 ->
    0.1793 (−10%), ep0 mIoU 73.87 -> 74.91. So NeCo does start lower, but only
    modestly; the 2 pp is built over training, not handed out at init. Consistent
    with §3.4's verdict that the loss-start probe is weak (there the sign was even
    reversed at +0.1–0.6% while mAP was flat).
    Not a feature-scale artifact: mean token L2 49.6 vs 52.9 (+7%).
  - **cross-head transfer**: dinov2 head on neco 71.39, neco head on dinov2 49.86
    (own: 79.65 / 81.62). Asymmetric -> NeCo is not a rescaling of DINOv2; a head
    trained on the sharper geometry does not survive the move back.
  - **option 4 (gradient cosine, 20 iters, bs 8, eval mode)**: DINOv2 −0.002 ± 0.088
    (noise +0.045), NeCo −0.015 ± 0.073 (noise −0.006); no block exceeds |0.05|.
    So the SSL gradient is *orthogonal* to the CE gradient for both -> gradient
    alignment is not the mechanism (null result, useful as a negative control).
    Norm ratio ‖g_ssl‖/‖g_ce‖ = 0.79 (DINOv2) vs 0.59 (NeCo): NeCo asks for less
    backbone movement per unit of downstream signal — same "head absorbs it"
    behaviour as in the ActionFormer encoder (§6.2).
  - **option 5 (patch kNN vs GT labels, 400 imgs, k=10)**: same-label fraction
    0.947 -> 0.973 (within), 0.918 -> 0.952 (cross); chance 0.656 / 0.513, so
    lift 0.846 -> 0.922 and 0.832 -> 0.901. k-means NMI 0.332 -> 0.369,
    ARI 0.044 -> 0.049. NeCo improves exactly the quantity its objective
    optimises, at a fixed k.
  - conclusion for the thesis: NeCo is *not* generically "too far from the
    downstream objective" — on dense prediction it helps by ~2 pp. The
    ActionFormer null result is a statement about applying it to a frozen,
    already-supervised feature encoder for sparse temporal labelling (no patch
    adjacency prior, aggregation rather than per-element labelling).
  - files: `neco_diag/{linear_probe,grad_alignment,neighbor_label_agreement,
    make_figs}.py`, results JSONs in `neco_diag/results/`, figure
    `neco_diag/figs/neco_diagnostics.png`; write-up `report.md` §7.
  - caveats: 1 seed per VOC run; option 5 on 400 images; option 4 can only
    resolve alignment >~0.05.
  - queue note: SLURM GPU jobs sat PENDING (Priority) for hours; ran on the idle
    login-node L4 instead (login-node zfs I/O was slow, ~3.7 min/epoch).

# 22-23/11
- found 3 problems in the SSL pipeline (details: CLAUDE.md / report.md header):
  NeCo references from the student instead of the teacher; EMA momentum 0.9995 left
  the from-scratch teacher (= transferred encoder) at its init (moved 0.08%); P2
  warm-started from raw weights but was evaluated on EMA heads (cost 0.37 alone).
  -> all SSL results before 22/11 are invalid. Fixed + reran (configs *_fixed*).
- corrected: P2 21.73 (vs 23.07); P3 SlowFast 23.82 +- 0.18 (3 seeds) vs 23.38 +- 0.75
  -> +0.44 n.s.; P3 VideoMAE 29.64 (temporal) / 29.22 (spatial) vs 29.60.
- why: SSL does train the encoder (4-5%) but moves timestep neighbourhoods AWAY from
  verbs (kNN verb lift .31 -> .26) and away from kitchens; worse first-epoch loss;
  erased by supervised training. spatial crops give the same drop -> not the views;
  kitchen probe -> not a scene shortcut. VOC control: same probes, opposite sign.
- paper rewritten from corrected results only (paper/main.pdf, 11 pp).
