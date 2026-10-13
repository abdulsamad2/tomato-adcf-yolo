import csv, glob, os
for f in sorted(glob.glob(r"C:\Users\absamad\cs\tomato-adcf-yolo\runs\*\seed*\results.csv")):
    rows = list(csv.DictReader(open(f)))
    fit = [(int(x["epoch"]), 0.1*float(x["metrics/mAP50(B)"]) + 0.9*float(x["metrics/mAP50-95(B)"])) for x in rows]
    best, bests = -1e9, []
    for e, v in fit:
        if v > best:
            best, _ = v, bests.append(e)
    gaps = [b - a for a, b in zip(bests, bests[1:])]
    d = os.path.dirname(f)
    name = os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d)
    done = os.path.exists(os.path.join(d, "train_done.json"))
    print(f"{name}|{len(fit)}|{bests[-1]}|{max(gaps) if gaps else 0}|{done}")
