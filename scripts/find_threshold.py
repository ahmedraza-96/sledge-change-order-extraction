"""Find the review threshold from labelled runs instead of choosing it by hand.

Fields with confidence below the threshold go to a human. The threshold trades errors caught
(recall) against fields sent to review, so the rule for picking it is fixed up front:

- every candidate threshold is scored with F2 (recall counted twice as heavily as precision),
  the usual choice when a missed error costs more than an extra review;
- the threshold is then checked on documents it was not picked from (leave one document out:
  pick on all the others, apply to the one left out), which is what to expect on new documents.

Usage:
    uv run python scripts/find_threshold.py RUNS_DIR:TRUTH_DIR [RUNS_DIR:TRUTH_DIR ...] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from co_extract.evaluate import FIELDS, field_correct, load_truths

GRID = [round(0.50 + i * 0.01, 2) for i in range(51)]  # 0.50 .. 1.00
BETA = 2.0


def load_points(pairs: list[str]) -> list[dict]:
    pts = []
    for pair in pairs:
        runs, truth_dir = pair.split(":", 1) if ":" in pair[2:] else (pair, "data/truth")
        truths = load_truths(Path(truth_dir))
        for p in sorted(Path(runs).glob("*.json")):
            if p.stem not in truths:
                continue
            r = json.loads(p.read_text())
            if any(i["code"] == "not_a_change_order" for i in r["issues"]):
                continue  # scores are pinned at 0: every field is reviewed whatever the threshold
            for n in FIELDS:
                pts.append({"doc": f"{Path(runs).name}/{p.stem}", "field": n, "conf": r["fields"][n]["confidence"],
                            "ok": field_correct(n, r["change_order"][n], truths[p.stem][n])})
    return pts


def at(pts: list[dict], th: float) -> dict:
    flagged = [p for p in pts if p["conf"] < th]
    wrong = sum(not p["ok"] for p in pts)
    caught = sum(not p["ok"] for p in flagged)
    recall = caught / wrong if wrong else 1.0
    precision = caught / len(flagged) if flagged else 0.0
    f = ((1 + BETA**2) * precision * recall / (BETA**2 * precision + recall)) if precision + recall else 0.0
    return {"th": th, "caught": caught, "wrong": wrong, "flagged": len(flagged), "n": len(pts),
            "recall": recall, "review_rate": len(flagged) / len(pts), "precision": precision, "f2": f}


def best(pts: list[dict]) -> float:
    # Highest F2; ties go to the lower threshold (fewer reviews for the same score).
    return max(GRID, key=lambda th: (round(at(pts, th)["f2"], 9), -th))


def leave_one_doc_out(pts: list[dict]) -> dict:
    docs = sorted({p["doc"] for p in pts})
    caught = wrong = flagged = 0
    picks = []
    for d in docs:
        th = best([p for p in pts if p["doc"] != d])
        held = [p for p in pts if p["doc"] == d]
        s = at(held, th)
        caught, wrong, flagged = caught + s["caught"], wrong + s["wrong"], flagged + s["flagged"]
        picks.append(th)
    return {"recall": caught / wrong if wrong else 1.0, "caught": caught, "wrong": wrong,
            "review_rate": flagged / len(pts), "picks": (min(picks), max(picks))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="RUNS_DIR[:TRUTH_DIR] (truth defaults to data/truth)")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    pts = load_points(args.pairs)
    th = best(pts)
    cv = leave_one_doc_out(pts)
    s = at(pts, th)
    out = ["# Review threshold test", "",
           f"{len({p['doc'] for p in pts})} documents, {len(pts)} fields, {s['wrong']} wrong. "
           f"Rule: highest F2 (recall weighted {BETA:g}x), checked leaving one document out.", "",
           f"**Chosen threshold: {th:.2f}**: catches {s['caught']}/{s['wrong']} errors ({s['recall']:.0%}), "
           f"sends {s['review_rate']:.0%} of fields to review.", "",
           f"On documents it was not picked from: catches {cv['caught']}/{cv['wrong']} ({cv['recall']:.0%}), "
           f"{cv['review_rate']:.0%} reviewed (threshold picked per fold: {cv['picks'][0]:.2f}-{cv['picks'][1]:.2f}).", "",
           "| below | errors caught | recall | fields reviewed | review rate | flagged that are wrong | F2 |",
           "|---|---|---|---|---|---|---|"]
    for t in sorted(set(GRID[::5]) | {th}):
        r = at(pts, t)
        out.append(f"| {t:.2f}{' <-- chosen' if t == th else ''} | {r['caught']}/{r['wrong']} | {r['recall']:.0%} | {r['flagged']} "
                   f"| {r['review_rate']:.0%} | {r['precision']:.0%} | {r['f2']:.3f} |")
    out += ["", "Wrong fields, lowest confidence first:", ""]
    out += [f"- {p['conf']:.3f} {p['doc']} `{p['field']}`" for p in sorted(pts, key=lambda p: p["conf"]) if not p["ok"]]
    text = "\n".join(out) + "\n"
    print(text)
    if args.report:
        args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
