# Clean-test pre-registration (one-time)

Frozen 2026-10-13 09:01:01. Status: **not run**.

## Test set
- 268 images, 3205 boxes, 261 families (clusters); sha256 `97873e6a1c1588e0e8d65acf42a3a0ac1b693721e964bb75afc562bf94bc74de`
- clean_test/verify_report.json (V1-V5 all pass)
- **History disclosure:** The clean test split HAS been scored before, for models outside this plan: the original adcf-run pipeline (results_test.json per run, results/tables_test.md, Sep 21 - Oct 3) and the Phase 2B single test evaluation (results/p2b_test_eval.json, Oct 5, 15 YOLO11s checkpoints). capacity_audit.py and res_audit.py later read those stored numbers. No model in this plan, and no protocol choice in Phases 4-17, was scored on or selected with the test split.

## Rows (seeds 0-4, step 24570, 512 px)
- **kd075**: PRIMARY: distilled 0.75x (Phase 17)
- **w075**: ordinary 0.75x (Phase 15)
- **w100**: from-scratch 1.00x (Phase 15)
- **r512**: pretrained 1.00x (Phase 4, Phase 17 teacher family)

## Checkpoints
| row | seed | sha256 | path |
|---|---|---|---|
| kd075 | 0 | `fe69a30160b7c6da` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p17\seed0\weights\ckpt_ep39_step24570.pt` |
| kd075 | 1 | `605ddb586e3689fe` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p17\seed1\weights\ckpt_ep39_step24570.pt` |
| kd075 | 2 | `24f93a6474d232d4` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p17\seed2\weights\ckpt_ep39_step24570.pt` |
| kd075 | 3 | `0255b0f7a7fd4729` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p17\seed3\weights\ckpt_ep39_step24570.pt` |
| kd075 | 4 | `51869df66f544427` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p17\seed4\weights\ckpt_ep39_step24570.pt` |
| w075 | 0 | `16f98d977da319bb` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w075\seed0\weights\ckpt_ep39_step24570.pt` |
| w075 | 1 | `a04196a4abc1d6d7` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w075\seed1\weights\ckpt_ep39_step24570.pt` |
| w075 | 2 | `fa61cd449f3a0c4f` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w075\seed2\weights\ckpt_ep39_step24570.pt` |
| w075 | 3 | `1679d9991505fb6b` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w075\seed3\weights\ckpt_ep39_step24570.pt` |
| w075 | 4 | `1cb03ae58e211434` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w075\seed4\weights\ckpt_ep39_step24570.pt` |
| w100 | 0 | `b325d0f642f295e2` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w100\seed0\weights\ckpt_ep39_step24570.pt` |
| w100 | 1 | `7c689358aab59bd4` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w100\seed1\weights\ckpt_ep39_step24570.pt` |
| w100 | 2 | `b1c9717825c10691` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w100\seed2\weights\ckpt_ep39_step24570.pt` |
| w100 | 3 | `408bbc7bd5c084e1` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w100\seed3\weights\ckpt_ep39_step24570.pt` |
| w100 | 4 | `5241474ace505d29` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p15\w100\seed4\weights\ckpt_ep39_step24570.pt` |
| r512 | 0 | `756f917bc7d27a9c` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p4\r512\seed0\weights\ckpt_ep39_step24570.pt` |
| r512 | 1 | `43498c5bf050f436` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p4\r512\seed1\weights\ckpt_ep39_step24570.pt` |
| r512 | 2 | `95fce76292108cc9` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p4\r512\seed2\weights\ckpt_ep39_step24570.pt` |
| r512 | 3 | `df8fcac48589bfc4` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p4\r512\seed3\weights\ckpt_ep39_step24570.pt` |
| r512 | 4 | `d1cd09c0ada38852` | `C:\Users\absamad\cs\tomato-adcf-yolo\runs_p4\r512\seed4\weights\ckpt_ep39_step24570.pt` |

## Settings
- imgsz: 512
- conf: 0.001
- nms_iou: 0.7
- max_det: 300
- top_per_image: 100
- tta: False
- scorer: pycocotools COCO AP@[.50:.95], maxDets 100, 8 categories (as Phases 9-17)
- decomposition: conf 0.25, greedy one-to-one class-agnostic match IoU 0.50 / strict 0.75 (Phase 12 rules)
- family_unit: split-manifest cluster (near-duplicate-merged photo group)

## Endpoints
- primary: six-class AP50-95 of the distilled 0.75x on the clean test (mean +/- sd over seeds 0-4)
- secondary: AP50
- secondary: AP75
- secondary: AR100
- secondary: localization recall @IoU 0.50
- secondary: localization recall @IoU 0.75
- secondary: conditional class correctness
- secondary: false positives per family
- secondary: APs, per-class AP (descriptive)

## Comparisons
paired-by-seed deltas (distilled minus each reference) with 95% t CI, and a 1000-resample family-cluster bootstrap of the seed-mean AP and loc-recall@0.50 deltas. Descriptive confirmation only: there is NO gate, and no result can change model selection.

## Rules
- No TTA (augment=False, as in every prior phase).
- No threshold tuning; one fixed set of constants.
- No class-specific settings.
- No test image is viewed; only image dimensions are read from headers.
- Nothing is printed or written until all 20 checkpoints are scored; one results file.
- No model selection or modification based on test results; the selection (distilled 0.75x) is locked.
- Runs once: the script refuses if results.json or RUN_LOCK exists.

## Static check of ct_eval.py
```
{
 "constants": {
  "CONF_AP": 0.001,
  "IOU_NMS": 0.7,
  "MAXDET": 300,
  "IMGSZ": 512,
  "TOPK": 100,
  "CONF": 0.25,
  "IOU_MATCH": 0.5,
  "IOU_STRICT": 0.75,
  "STEP": 24570
 },
 "n_predict_calls": 1,
 "predict_kwargs": {
  "imgsz": "IMGSZ",
  "conf": "CONF_AP",
  "iou": "IOU_NMS",
  "max_det": "MAXDET",
  "augment": "False",
  "device": "0",
  "verbose": "False"
 },
 "no_best_checkpoint": true,
 "no_conf_loop": true,
 "no_selection_tokens": true,
 "pass": true
}
```

Expected runtime: 5-15 (validation-split smoke test: 8 checkpoints + 20 resamples in 0.8 min; full run 20 checkpoints + 1000 resamples) min
