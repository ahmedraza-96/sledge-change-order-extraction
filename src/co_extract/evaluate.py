"""Score pipeline output against ground truth and check confidence calibration."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from .schema import FIELD_SPECS, FieldKind

FIELDS = [*FIELD_SPECS, "line_items"]
BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]


def _co_key(s: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", s.upper())
    return re.sub(r"^(PCO|CO|CHANGEORDER|NO)+", "", s)


def field_correct(name: str, predicted, truth) -> bool:
    if truth is None or predicted is None:
        return truth is None and predicted is None
    if name == "line_items":
        return sorted(li["amount"] for li in predicted) == sorted(li["amount"] for li in truth)
    if name == "change_order_number":
        return _co_key(str(predicted)) == _co_key(str(truth))
    if name == "description":
        return fuzz.token_set_ratio(str(predicted).lower(), str(truth).lower()) >= 55
    if FIELD_SPECS.get(name) == FieldKind.STR:
        return fuzz.token_set_ratio(str(predicted).lower(), str(truth).lower()) >= 85
    return predicted == truth


def score(results: dict[str, dict], truths: dict[str, dict]) -> dict:
    per_field = defaultdict(lambda: [0, 0])
    points: list[tuple[float, bool]] = []  # (confidence, correct)
    docs = []
    for stem, res in sorted(results.items()):
        truth = truths[stem]
        wrong, flagged = [], set(res["needs_review"])
        for name in FIELDS:
            ok = field_correct(name, res["change_order"][name], truth[name])
            per_field[name][0] += ok
            per_field[name][1] += 1
            points.append((res["fields"][name]["confidence"], ok))
            if not ok:
                wrong.append(name)
        produced = {i["code"] for i in res["issues"]}
        expected = set(truth.get("_expected_issues", []))
        docs.append({
            "doc": stem,
            "variant": truth.get("_variant", ""),
            "accuracy": 1 - len(wrong) / len(FIELDS),
            "wrong": wrong,
            "wrong_unflagged": [w for w in wrong if w not in flagged],
            "flagged": sorted(flagged),
            "doc_confidence": res["document_confidence"],
            "expected_issues_caught": sorted(expected & produced),
            "expected_issues_missed": sorted(expected - produced),
            "llm_calls": res["meta"]["llm_calls"],
            "tokens": res["meta"]["input_tokens"] + res["meta"]["output_tokens"],
            "seconds": res["meta"]["seconds"],
        })

    buckets = []
    ece, n = 0.0, len(points)
    for lo, hi in BUCKETS:
        pts = [(c, ok) for c, ok in points if lo <= c < hi]
        if not pts:
            buckets.append({"range": f"{lo:.1f}-{min(hi, 1):.1f}", "n": 0, "accuracy": None, "mean_conf": None})
            continue
        acc = sum(ok for _, ok in pts) / len(pts)
        mc = sum(c for c, _ in pts) / len(pts)
        ece += len(pts) / n * abs(acc - mc)
        buckets.append({"range": f"{lo:.1f}-{min(hi, 1):.1f}", "n": len(pts), "accuracy": acc, "mean_conf": mc})

    total_wrong = sum(len(d["wrong"]) for d in docs)
    total_flagged = sum(len(d["flagged"]) for d in docs)
    wrong_flagged = sum(len(d["wrong"]) - len(d["wrong_unflagged"]) for d in docs)
    return {
        "field_accuracy": sum(ok for _, ok in points) / n,
        "per_field": {k: v[0] / v[1] for k, v in per_field.items()},
        "docs": docs,
        "calibration": buckets,
        "ece": ece,
        "review_rate": total_flagged / n,
        "review_precision": wrong_flagged / total_flagged if total_flagged else None,
        "error_recall": wrong_flagged / total_wrong if total_wrong else None,
        "n_fields": n,
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_markdown(s: dict, model: str) -> str:
    out = [
        "# Evaluation report", "",
        f"Model: `{model}` · documents: {len(s['docs'])} · fields scored: {s['n_fields']}", "",
        "## Headline", "",
        "| metric | value |", "|---|---|",
        f"| field accuracy | {_pct(s['field_accuracy'])} |",
        f"| errors caught by review flags (recall) | {_pct(s['error_recall'])} |",
        f"| flagged fields that were actually wrong (precision) | {_pct(s['review_precision'])} |",
        f"| share of fields sent to review | {_pct(s['review_rate'])} |",
        f"| expected calibration error | {s['ece']:.3f} |", "",
        "## Calibration (all fields)", "",
        "| confidence | n | accuracy | mean confidence |", "|---|---|---|---|",
    ]
    for b in s["calibration"]:
        out.append(f"| {b['range']} | {b['n']} | {_pct(b['accuracy'])} | "
                   f"{'n/a' if b['mean_conf'] is None else f'{b['mean_conf']:.2f}'} |")
    out += ["", "## Per document", "",
            "| doc | variant | accuracy | doc conf | wrong (unflagged in **bold**) | expected issues caught / missed | calls | tokens | s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for d in s["docs"]:
        wrong = ", ".join(f"**{w}**" if w in d["wrong_unflagged"] else w for w in d["wrong"]) or "—"
        exp = f"{', '.join(d['expected_issues_caught']) or '—'} / {', '.join(d['expected_issues_missed']) or '—'}"
        out.append(f"| {d['doc']} | {d['variant']} | {_pct(d['accuracy'])} | {d['doc_confidence']:.2f} | {wrong} | {exp} "
                   f"| {d['llm_calls']} | {d['tokens']} | {d['seconds']} |")
    out += ["", "## Per field", "", "| field | accuracy |", "|---|---|"]
    out += [f"| {k} | {_pct(v)} |" for k, v in s["per_field"].items()]
    return "\n".join(out) + "\n"


def load_truths(truth_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(truth_dir.glob("*.json"))}
