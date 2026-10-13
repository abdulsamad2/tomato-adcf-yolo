"""Freeze the one-time clean-test plan: prereg.json / prereg.md / analysis_freeze.json (no test metric)."""
import ast, hashlib, json, sys, time
from pathlib import Path

B = Path(r"C:\Users\absamad\cs"); R = B / "tomato-adcf-yolo"; OUT = B / "clean_test"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ROOT = {"kd075": R / "runs_p17" / "seed{s}", "w075": R / "runs_p15" / "w075" / "seed{s}",
        "w100": R / "runs_p15" / "w100" / "seed{s}", "r512": R / "runs_p4" / "r512" / "seed{s}"}
SCRIPTS = ["ct_verify.py", "ct_eval.py", "ct_freeze.py"]


def static_check():
    """Confirm ct_eval.py has no threshold search / model selection: constants are the frozen ones, the predict
    call uses only those names, and no 'best'/'argmax-over-models' style selection exists."""
    src = (OUT / "ct_eval.py").read_text(encoding="utf-8"); tree = ast.parse(src)
    consts = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Tuple):
            for t, v in zip(n.targets[0].elts, n.value.elts):
                if isinstance(t, ast.Name) and isinstance(v, ast.Constant): consts[t.id] = v.value
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "predict"]
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords} if calls else {}
    low = src.lower()
    chk = {"constants": {k: consts.get(k) for k in ("CONF_AP", "IOU_NMS", "MAXDET", "IMGSZ", "TOPK", "CONF", "IOU_MATCH", "IOU_STRICT", "STEP")},
           "n_predict_calls": len(calls), "predict_kwargs": kw,
           "no_best_checkpoint": "best.pt" not in low and "best_fitness" not in low,
           "no_conf_loop": "for conf" not in low and "np.linspace(0.0" not in low and "thresholds" not in low,
           "no_selection_tokens": all(t not in low for t in ("select(", "argmax(table", "max(rows", "sorted(rows", "choose"))}
    chk["pass"] = chk["constants"] == {"CONF_AP": 0.001, "IOU_NMS": 0.7, "MAXDET": 300, "IMGSZ": 512, "TOPK": 100,
                                       "CONF": 0.25, "IOU_MATCH": 0.5, "IOU_STRICT": 0.75, "STEP": 24570} \
        and len(calls) == 1 and kw.get("conf") == "CONF_AP" \
        and kw.get("iou") == "IOU_NMS" and kw.get("augment") == "False" and chk["no_best_checkpoint"] \
        and chk["no_conf_loop"] and chk["no_selection_tokens"]
    return chk


def build_prereg():
    vr = json.loads((OUT / "verify_report.json").read_text())
    assert vr["all_pass"], "verification did not pass"
    ck = {}
    for m, root in ROOT.items():
        ck[m] = {}
        for s in range(5):
            p = Path(str(root).format(s=s)) / "weights" / "ckpt_ep39_step24570.pt"
            ck[m][str(s)] = {"path": str(p), "sha256": sha(p)}
    return {
     "title": "One-time clean-test evaluation", "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
     "status": "frozen; NOT run",
     "test_set": {"split": "data/tomato_village images/test + labels/test", "n_images": 268, "n_boxes": vr["n_boxes"],
                  "n_clusters_families": vr["V2_manifest_disjoint"]["test_clusters"], "sha256": vr["test_set_sha256"],
                  "verification": "clean_test/verify_report.json (V1-V5 all pass)",
                  "history_disclosure": vr["V5_prior_access"]["statement"]},
     "rows": {"kd075": "PRIMARY: distilled 0.75x (Phase 17)", "w075": "ordinary 0.75x (Phase 15)",
              "w100": "from-scratch 1.00x (Phase 15)", "r512": "pretrained 1.00x (Phase 4, Phase 17 teacher family)"},
     "seeds": [0, 1, 2, 3, 4], "checkpoint_step": 24570, "checkpoints": ck,
     "settings": {"imgsz": 512, "conf": 0.001, "nms_iou": 0.7, "max_det": 300, "top_per_image": 100, "tta": False,
                  "scorer": "pycocotools COCO AP@[.50:.95], maxDets 100, 8 categories (as Phases 9-17)",
                  "decomposition": "conf 0.25, greedy one-to-one class-agnostic match IoU 0.50 / strict 0.75 (Phase 12 rules)",
                  "family_unit": "split-manifest cluster (near-duplicate-merged photo group)"},
     "primary_endpoint": "six-class AP50-95 of the distilled 0.75x on the clean test (mean +/- sd over seeds 0-4)",
     "secondary_endpoints": ["AP50", "AP75", "AR100", "localization recall @IoU 0.50", "localization recall @IoU 0.75",
                             "conditional class correctness", "false positives per family", "APs, per-class AP (descriptive)"],
     "comparisons": ("paired-by-seed deltas (distilled minus each reference) with 95% t CI, and a 1000-resample "
                     "family-cluster bootstrap of the seed-mean AP and loc-recall@0.50 deltas. Descriptive "
                     "confirmation only: there is NO gate, and no result can change model selection."),
     "rules": ["No TTA (augment=False, as in every prior phase).", "No threshold tuning; one fixed set of constants.",
               "No class-specific settings.", "No test image is viewed; only image dimensions are read from headers.",
               "Nothing is printed or written until all 20 checkpoints are scored; one results file.",
               "No model selection or modification based on test results; the selection (distilled 0.75x) is locked.",
               "Runs once: the script refuses if results.json or RUN_LOCK exists."],
     "expected_runtime_minutes": "5-15 (validation-split smoke test: 8 checkpoints + 20 resamples in 0.8 min; "
                                 "full run 20 checkpoints + 1000 resamples)",
     "outputs": ["clean_test/results.json", "tomato-adcf-yolo/results/clean_test_table.csv",
                 "tomato-adcf-yolo/results/clean_test_table.md"]}


def main():
    for f in ("prereg.json", "analysis_freeze.json", "results.json", "RUN_LOCK"):
        if (OUT / f).exists(): sys.exit("ABORT: %s exists" % f)
    P = build_prereg(); chk = static_check()
    if not chk["pass"]: sys.exit("ABORT: static check failed " + json.dumps(chk))
    P["static_check_of_ct_eval"] = chk
    json.dump(P, open(OUT / "prereg.json", "w"), indent=1)
    md = ["# Clean-test pre-registration (one-time)", "", "Frozen %s. Status: **not run**." % P["frozen_at"], "",
          "## Test set", "- %d images, %d boxes, %d families (clusters); sha256 `%s`" % (268, P["test_set"]["n_boxes"],
          P["test_set"]["n_clusters_families"], P["test_set"]["sha256"]), "- " + P["test_set"]["verification"],
          "- **History disclosure:** " + P["test_set"]["history_disclosure"], "", "## Rows (seeds 0-4, step 24570, 512 px)"]
    md += ["- **%s**: %s" % (k, v) for k, v in P["rows"].items()]
    md += ["", "## Checkpoints", "| row | seed | sha256 | path |", "|---|---|---|---|"]
    md += ["| %s | %s | `%s` | `%s` |" % (m, s, c["sha256"][:16], c["path"]) for m, d in P["checkpoints"].items() for s, c in d.items()]
    md += ["", "## Settings"] + ["- %s: %s" % (k, v) for k, v in P["settings"].items()]
    md += ["", "## Endpoints", "- primary: " + P["primary_endpoint"]] + ["- secondary: " + x for x in P["secondary_endpoints"]]
    md += ["", "## Comparisons", P["comparisons"], "", "## Rules"] + ["- " + x for x in P["rules"]]
    md += ["", "## Static check of ct_eval.py", "```", json.dumps(chk, indent=1), "```", "",
           "Expected runtime: " + P["expected_runtime_minutes"] + " min", ""]
    (OUT / "prereg.md").write_text("\n".join(md), encoding="utf-8")
    AF = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "hashed before the clean test is run",
          "prereg_sha256": sha(OUT / "prereg.json"), "verify_report_sha256": sha(OUT / "verify_report.json"),
          "script_sha256": {s: sha(OUT / s) for s in SCRIPTS},
          "checkpoint_sha256": {"%s_seed%s" % (m, s): c["sha256"] for m, d in P["checkpoints"].items() for s, c in d.items()},
          "smoke_test": "ct_eval.py logic exercised on the VALIDATION split via a scratch copy before freezing (no test access)"}
    json.dump(AF, open(OUT / "analysis_freeze.json", "w"), indent=1)
    print("FROZEN", P["frozen_at"]); print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
