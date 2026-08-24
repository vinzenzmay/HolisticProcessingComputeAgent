"""Side-by-side of the edit-arm result JSONs, plus per-task success."""

import json
import sys
from pathlib import Path

paths = [Path(a) for a in sys.argv[1:]]
blobs = [json.loads(p.read_text()) for p in paths]
runs = [b["summary"] for b in blobs]
detail = [b["runs"] for b in blobs]

KEYS = [
    ("success_rate", "success", "{:.3f}"),
    ("mean_failed_edits", "failed edits/run", "{:.3f}"),
    ("mean_tool_errors", "tool errors/run", "{:.3f}"),
    ("mean_tool_calls", "tool calls/task", "{:.3f}"),
    ("total_completion_tokens", "completion tokens", "{:,}"),
    ("total_wall_s", "wall seconds", "{:.0f}"),
]

labels = [r.get("label", p.stem) for r, p in zip(runs, paths)]
n = [r.get("n_runs", 0) for r in runs]
w = 22
print("metric".ljust(w) + "".join(f"{lab:>18}" for lab in labels))
print("-" * (w + 18 * len(labels)))
print("n".ljust(w) + "".join(f"{x:>18}" for x in n))
for key, name, fmt in KEYS:
    row = name.ljust(w)
    for r in runs:
        v = r.get(key)
        row += f"{(fmt.format(v) if v is not None else '-'):>18}"
    print(row)

# Per-task success, to see which shapes move rather than only the average.
print("\nper-task success")
per = []
for r in detail:
    d = {}
    for rec in r:
        d.setdefault(rec["task"], []).append(bool(rec.get("success")))
    per.append(d)
tasks = sorted({t for d in per for t in d})
print("task".ljust(w) + "".join(f"{lab:>18}" for lab in labels))
for t in tasks:
    row = t[:w - 1].ljust(w)
    flag = ""
    rates = []
    for d in per:
        got = d.get(t, [])
        rate = sum(got) / len(got) if got else None
        rates.append(rate)
        row += f"{(f'{rate:.2f} ({len(got)})' if rate is not None else '-'):>18}"
    known = [x for x in rates if x is not None]
    if known and max(known) - min(known) >= 0.08:
        flag = "   <-- moves"
    print(row + flag)

print("\nper-task tool calls / failed edits")
print("task".ljust(w) + "".join(f"{lab:>18}" for lab in labels))
for t in tasks:
    row = t[:w - 1].ljust(w)
    for d in detail:
        recs = [x for x in d if x["task"] == t]
        if not recs:
            row += f"{'-':>18}"
            continue
        calls = sum(x["tool_calls"] for x in recs) / len(recs)
        fails = sum(x["failed_edits"] for x in recs) / len(recs)
        row += f"{f'{calls:.2f} / {fails:.2f}':>18}"
    print(row)

print("\nerrors seen")
for lab, d in zip(labels, detail):
    errs = {}
    for x in d:
        if x.get("error"):
            errs[x["error"][:70]] = errs.get(x["error"][:70], 0) + 1
    print(f"  {lab}: {errs or 'none'}")
