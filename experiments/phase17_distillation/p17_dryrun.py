"""Phase 17 dry run (BEFORE pre-registration and before any Phase 17 training step that is kept).

Three 20-batch runs on seed 0, written to phase17/dryrun (never to runs_p17):
  A  ordinary Phase 15 0.75x trainer (no KD code path at all)
  B  KD trainer with lam_ld = lam_at = 0 but the teacher loaded and forwarded every batch (probe)
     -> must reproduce A's per-batch training loss (KD disabled == ordinary loss path)
  C  KD trainer with the final coefficients -> KD terms finite, shapes as expected
Coefficient rule fixed before this script runs (no evaluation data used):
  lam_ld = 1.0, T_ld = 1.0 (soft-target analogue of the existing DFL term, at the same hyp.dfl gain)
  lam_at = mean_{first 10 batches of B} (hyp.dfl * L_LD) / mean_{same batches} (L_AT), rounded to 3 sig. figs,
           i.e. the two KD terms start at equal magnitude on the ordinary trajectory. Training images only.
"""
import hashlib, json, math, sys, time
from pathlib import Path
BASE = Path(r"C:\Users\absamad\cs"); REPO = BASE / "tomato-adcf-yolo"
sys.path.insert(0, str(BASE))
OUT = BASE / "phase17"; OUT.mkdir(exist_ok=True); DRY = OUT / "dryrun"
TEACHER = REPO / "runs_p4" / "r512" / "seed0" / "weights" / "ckpt_ep39_step24570.pt"
K, NCAL = 20, 10
assert not (OUT / "p17_prereg.json").exists(), "already frozen"


def main():
    import p17_run
    t0 = time.time()
    tsha = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
    rep = {"teacher": str(TEACHER), "teacher_sha256": tsha, "batches": K, "seed": 0}
    A, _ = p17_run.train(0, DRY / "A_ordinary", kd=None, stop_after=K, record={})
    B, _ = p17_run.train(0, DRY / "B_kd_zero", stop_after=K, record={},
                         kd={"teacher": str(TEACHER), "teacher_sha256": tsha, "lam_ld": 0.0, "lam_at": 0.0,
                             "t_ld": 1.0, "probe": True})
    la, lb = A["batch_loss"], B["batch_loss"]
    rel = max(abs(x - y) / max(abs(x), 1e-9) for x, y in zip(la, lb))
    rep["A_vs_B"] = {"A_batch_loss": la, "B_batch_loss": lb, "max_rel_diff": rel,
                     "init_sha_equal": A["init_sd_sha256"] == B["init_sd_sha256"], "pass": rel < 1e-4}
    import yaml
    dfl_gain = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text()).get("train", {}).get("dfl", 1.5)
    cal = B["kd_batch_log"][:NCAL]
    ld_m = sum(c[1] for c in cal) / NCAL; at_m = sum(c[2] for c in cal) / NCAL
    raw = dfl_gain * ld_m / at_m
    lam_at = float("%.3g" % raw)
    rep["calibration"] = {"dfl_gain": dfl_gain, "mean_LD_first10": ld_m, "mean_AT_first10": at_m,
                          "lam_at_raw": raw, "lam_at": lam_at, "lam_ld": 1.0, "T_ld": 1.0,
                          "det_loss_first10": sum(c[0] for c in cal) / NCAL}
    C, _ = p17_run.train(0, DRY / "C_kd_final", stop_after=K, record={},
                         kd={"teacher": str(TEACHER), "teacher_sha256": tsha, "lam_ld": 1.0, "lam_at": lam_at,
                             "t_ld": 1.0, "probe": False})
    fin = all(math.isfinite(x) for x in C["batch_loss"]) and all(math.isfinite(v) for c in C["kd_batch_log"] for v in c)
    kd_share = [(dfl_gain * ld + lam_at * at) / (det + dfl_gain * ld + lam_at * at) for det, ld, at in C["kd_batch_log"]]
    rep["C_final"] = {"batch_loss": C["batch_loss"], "kd_batch_log": C["kd_batch_log"], "finite": fin,
                      "kd_share_of_total_first": kd_share[0], "kd_share_of_total_last": kd_share[-1],
                      "init_sha_equal_A": C["init_sd_sha256"] == A["init_sd_sha256"]}
    rep["shapes"] = B["kd_shapes"]
    rep["student"] = {"params": A["params"], "first_conv_shape": A["first_conv_shape"],
                      "first_conv_equals_pretrained": A["first_conv_equals_pretrained"],
                      "init_sd_sha256": A["init_sd_sha256"], "schedule": A["schedule"], "mismatch": A["mismatch"]}
    rep["teacher_checks"] = {k: B[k] for k in ("teacher_sha256", "teacher_any_requires_grad", "teacher_training_mode",
                                               "teacher_params")}
    rep["all_pass"] = bool(rep["A_vs_B"]["pass"] and rep["A_vs_B"]["init_sha_equal"] and fin
                           and rep["C_final"]["init_sha_equal_A"] and A["params"] == 1565628
                           and not A["first_conv_equals_pretrained"] and not A["mismatch"]
                           and not B["teacher_any_requires_grad"] and not B["teacher_training_mode"]
                           and B["teacher_params"] == 2591400)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(rep, open(OUT / "p17_dryrun_report.json", "w"), indent=1)
    print(json.dumps({"A_vs_B_max_rel_diff": rel, "init_equal": rep["A_vs_B"]["init_sha_equal"],
                      "calibration": rep["calibration"], "finite": fin, "kd_share_first_last": [kd_share[0], kd_share[-1]],
                      "shapes": rep["shapes"], "teacher": rep["teacher_checks"], "student": {k: rep["student"][k] for k in
                      ("params", "first_conv_equals_pretrained", "first_conv_shape")}, "all_pass": rep["all_pass"]}, indent=1))
    print("DRYRUNDONE")


if __name__ == "__main__":
    main()
