# Phase 17 pre-registration: single-configuration localization distillation

Frozen 2026-10-12 23:51:41, before any kept Phase 17 training step. Machine-readable: `p17_prereg.json`.

## Question
Can knowledge distillation from the pretrained 1.00x teacher recover independent-source (arm B) accuracy in the Phase 15 0.75x student without changing its inference architecture?

## Context
- Phase 15 selected 0.75x (1,565,628 params, 2.707 GFLOPs) as equivalent to 1.00x within +/-1.5 AP.
- Phase 16A LPWA-50 failed its AP gate; architecture work is closed.
- Exploratory: COCO pretraining adds +3.9 AP on arm B vs +15.1 on exposed arm A.
- Deficits are dominated by localization; conditional class correctness stays ~97%.

## Teacher
- checkpoint `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p4\r512\seed0\weights\ckpt_ep39_step24570.pt`
- sha256 `756f917bc7d27a9cf599404a3de0d606974e8744adbc1987481d2a4d3aeec3a5`
- pretrained (COCO-init) YOLO11n 1.00x, runs_p4/r512 seed 0, fixed 24570-update checkpoint (not best-val); the model used in the exploratory pretraining analysis; 2591400 params; eval mode, requires_grad False, torch.no_grad

## Student
- `C:\Users\absamad\cs\widths\yolo11n_w0p75.yaml` (sha256 `157eb1175501af4f`)
- 1565628 params, 2.707 GFLOPs @512, no architecture change
- init: random initialisation exactly as Phase 15 w075 (yaml-built, pretrained=False, Ultralytics deterministic seeding with the run seed); the dry run verified identical initial state-dict hashes for the ordinary and KD trainers and that the first conv is not yolo11n.pt

## Training
identical to runs_p15/w075; the ONLY difference is the added KD loss; seeds 0-4, one at a time; 24,570 updates; checkpoints {'epoch10': 6300, 'epoch19': 11970, 'epoch29': 18270, 'epoch39': 24570}

## Distillation (one configuration)
- L = L_box + L_cls + L_dfl  +  lam_ld * hyp.dfl * L_LD  +  lam_at * L_AT;  L_LD = mean over the student's TaskAlignedAssigner foreground anchors and the 4 box sides of KL( softmax(t/T) || softmax(s/T) ) * T^2 over the 16 DFL bins;  L_AT = sum over P3,P4 of mean_b || q_S - q_T ||_2^2 with q = vec(mean_c F^2)/||vec(mean_c F^2)||_2;  the total is multiplied by the batch size (Ultralytics convention)
- lam_ld = 1.0, T = 1.0, hyp.dfl = 1.5, lam_at = 1.25
- lam_at rule: lam_at = mean_{first 10 batches, ordinary trajectory, seed 0}(hyp.dfl*L_LD) / mean(L_AT), 3 significant figures; computed on training images only, before pre-registration
- class-logit KD: none; trainable params added: 0
- hook points: Detect training output dict {boxes, scores, feats}: student box logits and neck P3/P4 feats are read directly (no hooks); teacher run in eval mode returns (y, dict) and the dict is used; assigner foreground mask from v8DetectionLoss.get_assigned_targets_and_loss
- not the built-in: Ultralytics 8.4.155 ships DistillationModel (distill_model=...): P3/P4/P5 feature imitation with trainable projector MLPs weighted by teacher class scores, no box-distribution term. It is not localization-focused and adds P5, so it does not match this phase's design.

## Dry run
- KD off reproduces the ordinary loss: max rel diff 0 over 20 batches
- shapes: {"student_boxes": [16, 64, 5376], "teacher_boxes": [16, 64, 5376], "student_feats": [[16, 48, 64, 64], [16, 96, 32, 32], [16, 192, 16, 16]], "teacher_feats": [[16, 64, 64, 64], [16, 128, 32, 32], [16, 256, 16, 16]], "fg_anchors_first_batch": 2574}

## Primary
- endpoint: six-class AP50-95 on arm B (213 independent families) at exactly 24570 updates
- comparison: kd075 vs ordinary w075, paired by seed
- statistic: Delta_i = AP_B(kd075, i) - AP_B(w075, i), i = 0..4
- gate_AP: 0.75
- rule: PASS iff mean Delta >= +0.75 AP AND the resource/architecture-identity gate holds
- reported: paired 95% t CI (df 4), per-seed deltas; significance not required

## Mechanism gate
mean Delta localization recall @IoU .50 on arm B > 0 (expected direction). Reported as mechanistic corroboration; not part of PASS; no significance test.

## Resource gate
per seed: params kd == base == 1,565,628; GFLOPs@512 equal; identical state-dict keys and shapes; checkpoint file size within 1%. Latency (GPU b1/b16, CPU b1) benchmarked on idle hardware; no speed-up is claimed because the inference graph is unchanged.

## Secondary
- AP50, AP75, APs, AR100, class-agnostic AP, locR50, locR75, conditional class correctness, FP/family
- arm A and arm B; exposed-minus-independent interaction (paired t and family bootstrap 1000)
- all of the above at 18270
- vs 1.00x (Phase 15 one-sided margin -1.5) and vs the pretrained teacher family (context)
- fraction of the pretrained-vs-0.75x gap recovered
- convergence (Phase 15 rule)

## Stop rules
- **fail**: mean Delta AP_B < +0.75: Phase 17 FAILS. No second KD method, no alternate teacher, no alternate loss weights, no architecture modification, no rescue experiment.
- **pass**: mean Delta AP_B >= +0.75 (and resource gate holds): Phase 17 PASSES; distilled 0.75x becomes the candidate deployment model; no further KD search.

## Integrity
- The clean test split is not inspected during development.
- The margin and all rules are not changed after any result is seen.
- Best-validation checkpoints are never the primary outcome; the primary is the fixed 24570-update checkpoint.
- No edge-device speed claim is made from desktop GPU timings.
- No Phase 17 AP is inspected during training (curves log losses and clean-val only; not read until post).
- Coefficients are frozen; none is tuned on arm A, arm B or any Phase 17 result.
- Clean test is opened once only, after Phase 17 finishes and final model selection is locked.

## Expected compute
{"training_hours": "3.0-3.5 (5 x ~36-42 min; teacher forward adds a few %)", "post_hours": "~1.0 (scoring 20 checkpoints ~35 min, bench ~5 min, stats ~25 min)", "total_wall_hours": "4.0-4.5"}
