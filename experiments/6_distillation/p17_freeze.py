"""Phase 17 step 0: freeze the single-configuration distillation experiment BEFORE any kept training step.

Writes phase17/p17_prereg.json + .md and phase17/p17_analysis_freeze.json (sha256 of every Phase 17 script).
ABORTS if already frozen, if runs_p17 holds any checkpoint, or if the dry run did not pass every check.
"""
import hashlib, json, sys, time
from pathlib import Path

BASE = Path(r"C:\Users\absamad\cs"); REPO = BASE / "tomato-adcf-yolo"; OUT = BASE / "phase17"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SCRIPTS = ["p17_kd.py", "p17_run.py", "p17_dryrun.py", "p17_freeze.py", "p17_eval.py", "p17_stats.py",
           "p17_bench.py", "p17_queue.ps1", "p17_post.ps1"]
for f in ("p17_prereg.json", "p17_analysis_freeze.json"):
    if (OUT / f).exists(): sys.exit("ABORT: %s exists" % f)
if (REPO / "runs_p17").exists() and any((REPO / "runs_p17").rglob("*.pt")): sys.exit("ABORT: runs_p17 has checkpoints")
D = json.loads((OUT / "p17_dryrun_report.json").read_text())
if not D["all_pass"]: sys.exit("ABORT: dry run did not pass")
C = D["calibration"]
TEACHER = REPO / "runs_p4" / "r512" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"
assert sha(TEACHER) == D["teacher_sha256"]
p15 = json.loads((BASE / "phase15" / "p15_prereg.json").read_text())

R = {
 "phase": 17, "title": "Single-configuration localization-focused distillation into the selected 0.75x model",
 "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "question": ("Can knowledge distillation from the pretrained 1.00x teacher recover independent-source (arm B) "
              "accuracy in the Phase 15 0.75x student without changing its inference architecture?"),
 "context": ["Phase 15 selected 0.75x (1,565,628 params, 2.707 GFLOPs) as equivalent to 1.00x within +/-1.5 AP.",
             "Phase 16A LPWA-50 failed its AP gate; architecture work is closed.",
             "Exploratory: COCO pretraining adds +3.9 AP on arm B vs +15.1 on exposed arm A.",
             "Deficits are dominated by localization; conditional class correctness stays ~97%."],
 "teacher": {"checkpoint": str(TEACHER), "sha256": D["teacher_sha256"],
             "description": "pretrained (COCO-init) YOLO11n 1.00x, runs_p4/r512 seed 0, fixed 24570-update checkpoint "
                            "(not best-val); the model used in the exploratory pretraining analysis",
             "params": D["teacher_checks"]["teacher_params"], "frozen": "eval mode, requires_grad False, torch.no_grad",
             "single_teacher_for_all_seeds": True},
 "student": {"yaml": str(BASE / "widths" / "yolo11n_w0p75.yaml"), "yaml_sha256": sha(BASE / "widths" / "yolo11n_w0p75.yaml"),
             "params": 1565628, "GFLOPs_512": 2.707, "imgsz": 512,
             "init_policy": ("random initialisation exactly as Phase 15 w075 (yaml-built, pretrained=False, Ultralytics "
                             "deterministic seeding with the run seed); the dry run verified identical initial state-dict "
                             "hashes for the ordinary and KD trainers and that the first conv is not yolo11n.pt"),
             "init_sd_sha256_seed0": D["student"]["init_sd_sha256"], "architecture_change": "none"},
 "baseline": {"runs": "runs_p15/w075/seed0-4 (ordinary, non-distilled, already trained and scored)",
              "scores": "phase15/eval_rows.json (unchanged)"},
 "training": dict(p15["training"], note="identical to runs_p15/w075; the ONLY difference is the added KD loss"),
 "seeds": [0, 1, 2, 3, 4], "run_order": "one seed at a time, 0..4",
 "checkpoints": p15["checkpoints"], "primary_checkpoint": 24570, "secondary_checkpoint": 18270,
 "distillation": {
   "formula": ("L = L_box + L_cls + L_dfl  +  lam_ld * hyp.dfl * L_LD  +  lam_at * L_AT;  "
               "L_LD = mean over the student's TaskAlignedAssigner foreground anchors and the 4 box sides of "
               "KL( softmax(t/T) || softmax(s/T) ) * T^2 over the 16 DFL bins;  "
               "L_AT = sum over P3,P4 of mean_b || q_S - q_T ||_2^2 with q = vec(mean_c F^2)/||vec(mean_c F^2)||_2;  "
               "the total is multiplied by the batch size (Ultralytics convention)"),
   "lam_ld": 1.0, "T_ld": 1.0, "hyp_dfl": C["dfl_gain"], "lam_at": C["lam_at"],
   "lam_at_rule": ("lam_at = mean_{first 10 batches, ordinary trajectory, seed 0}(hyp.dfl*L_LD) / mean(L_AT), "
                   "3 significant figures; computed on training images only, before pre-registration"),
   "lam_at_calibration": C,
   "class_logit_kd": "none",
   "trainable_params_added": 0,
   "hook_points": ("Detect training output dict {boxes, scores, feats}: student box logits and neck P3/P4 feats are "
                   "read directly (no hooks); teacher run in eval mode returns (y, dict) and the dict is used; "
                   "assigner foreground mask from v8DetectionLoss.get_assigned_targets_and_loss"),
   "why_not_builtin": ("Ultralytics 8.4.155 ships DistillationModel (distill_model=...): P3/P4/P5 feature imitation "
                       "with trainable projector MLPs weighted by teacher class scores, no box-distribution term. It "
                       "is not localization-focused and adds P5, so it does not match this phase's design."),
   "kd_share_of_total_loss_dry_run": [D["C_final"]["kd_share_of_total_first"], D["C_final"]["kd_share_of_total_last"]]},
 "dry_run": {"report": str(OUT / "p17_dryrun_report.json"), "report_sha256": sha(OUT / "p17_dryrun_report.json"),
             "kd_off_reproduces_ordinary_loss_max_rel_diff": D["A_vs_B"]["max_rel_diff"],
             "shapes": D["shapes"], "all_pass": True},
 "evaluation": p15["evaluation"],
 "primary": {"endpoint": "six-class AP50-95 on arm B (213 independent families) at exactly 24570 updates",
             "comparison": "kd075 vs ordinary w075, paired by seed",
             "statistic": "Delta_i = AP_B(kd075, i) - AP_B(w075, i), i = 0..4",
             "gate_AP": 0.75,
             "rule": "PASS iff mean Delta >= +0.75 AP AND the resource/architecture-identity gate holds",
             "reported": "paired 95% t CI (df 4), per-seed deltas; significance not required"},
 "mechanism_gate": ("mean Delta localization recall @IoU .50 on arm B > 0 (expected direction). Reported as "
                    "mechanistic corroboration; not part of PASS; no significance test."),
 "resource_gate": ("per seed: params kd == base == 1,565,628; GFLOPs@512 equal; identical state-dict keys and "
                   "shapes; checkpoint file size within 1%. Latency (GPU b1/b16, CPU b1) benchmarked on idle "
                   "hardware; no speed-up is claimed because the inference graph is unchanged."),
 "secondary": ["AP50, AP75, APs, AR100, class-agnostic AP, locR50, locR75, conditional class correctness, FP/family",
               "arm A and arm B; exposed-minus-independent interaction (paired t and family bootstrap 1000)",
               "all of the above at 18270",
               "vs 1.00x (Phase 15 one-sided margin -1.5) and vs the pretrained teacher family (context)",
               "fraction of the pretrained-vs-0.75x gap recovered", "convergence (Phase 15 rule)"],
 "stop_rules": {
   "fail": ("mean Delta AP_B < +0.75: Phase 17 FAILS. No second KD method, no alternate teacher, no alternate loss "
            "weights, no architecture modification, no rescue experiment."),
   "pass": ("mean Delta AP_B >= +0.75 (and resource gate holds): Phase 17 PASSES; distilled 0.75x becomes the "
            "candidate deployment model; no further KD search.")},
 "integrity_rules": p15["integrity_rules"] + [
   "No Phase 17 AP is inspected during training (curves log losses and clean-val only; not read until post).",
   "Coefficients are frozen; none is tuned on arm A, arm B or any Phase 17 result.",
   "Clean test is opened once only, after Phase 17 finishes and final model selection is locked."],
 "expected_compute": {"training_hours": "3.0-3.5 (5 x ~36-42 min; teacher forward adds a few %)",
                      "post_hours": "~1.0 (scoring 20 checkpoints ~35 min, bench ~5 min, stats ~25 min)",
                      "total_wall_hours": "4.0-4.5"},
 "outputs": {"runs": "runs_p17/seed0-4", "results": ["results/p17_checkpoint_results.csv", "results/p17_stats.json",
             "results/p17_efficiency.json"], "logs": ["logs/p17.log", "logs/p17post.log"]},
 "script_sha256": {s: sha(BASE / s) for s in SCRIPTS}}
json.dump(R, open(OUT / "p17_prereg.json", "w"), indent=1)

K = R["distillation"]
md = ["# Phase 17 pre-registration: single-configuration localization distillation", "",
      "Frozen %s, before any kept Phase 17 training step. Machine-readable: `p17_prereg.json`." % R["frozen_at"], "",
      "## Question", R["question"], "", "## Context"] + ["- " + c for c in R["context"]] + [
      "", "## Teacher", "- checkpoint `%s`" % R["teacher"]["checkpoint"], "- sha256 `%s`" % R["teacher"]["sha256"],
      "- " + R["teacher"]["description"] + "; %d params; %s" % (R["teacher"]["params"], R["teacher"]["frozen"]),
      "", "## Student", "- `%s` (sha256 `%s`)" % (R["student"]["yaml"], R["student"]["yaml_sha256"][:16]),
      "- %d params, %.3f GFLOPs @512, no architecture change" % (R["student"]["params"], R["student"]["GFLOPs_512"]),
      "- init: " + R["student"]["init_policy"], "", "## Training", R["training"]["note"] + "; seeds 0-4, one at a time; "
      "24,570 updates; checkpoints " + str(R["checkpoints"]), "", "## Distillation (one configuration)",
      "- " + K["formula"], "- lam_ld = %.1f, T = %.1f, hyp.dfl = %.1f, lam_at = %g" % (K["lam_ld"], K["T_ld"], K["hyp_dfl"], K["lam_at"]),
      "- lam_at rule: " + K["lam_at_rule"], "- class-logit KD: none; trainable params added: 0",
      "- hook points: " + K["hook_points"], "- not the built-in: " + K["why_not_builtin"],
      "", "## Dry run", "- KD off reproduces the ordinary loss: max rel diff %g over 20 batches" % R["dry_run"]["kd_off_reproduces_ordinary_loss_max_rel_diff"],
      "- shapes: " + json.dumps(R["dry_run"]["shapes"]), "", "## Primary"] + \
     ["- %s: %s" % (k, v) for k, v in R["primary"].items()] + ["", "## Mechanism gate", R["mechanism_gate"],
      "", "## Resource gate", R["resource_gate"], "", "## Secondary"] + ["- " + x for x in R["secondary"]] + \
     ["", "## Stop rules"] + ["- **%s**: %s" % (k, v) for k, v in R["stop_rules"].items()] + \
     ["", "## Integrity"] + ["- " + x for x in R["integrity_rules"]] + \
     ["", "## Expected compute", json.dumps(R["expected_compute"]), ""]
(OUT / "p17_prereg.md").write_text("\n".join(md), encoding="utf-8")

AF = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
      "note": "all Phase 17 scripts hashed before any Phase 17 training step that is kept and before any Phase 17 metric",
      "prereg_sha256": sha(OUT / "p17_prereg.json"), "dryrun_report_sha256": sha(OUT / "p17_dryrun_report.json"),
      "sha256": {s: sha(BASE / s) for s in SCRIPTS},
      "self_test": "p17_stats.py recovered a planted +1.0 AP effect (+0.85 over 5 seeds) on synthetic rows; both branches resolve"}
json.dump(AF, open(OUT / "p17_analysis_freeze.json", "w"), indent=1)
print("FROZEN", R["frozen_at"], "| lam_at", K["lam_at"])
