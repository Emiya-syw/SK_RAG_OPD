#!/usr/bin/env python3
"""Render veRL file-logger JSONL into a compact SVG and selected-metrics CSV.

Uses only the Python standard library so the plot can also be regenerated away
from the training environment: python scripts/plot_train_metrics.py metrics.jsonl.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


PANELS = (
    ("Loss", ("actor/distillation/loss", "actor/loss", "actor/distillation/forward_component", "actor/distillation/sampled_reverse_kl")),
    ("Entropy (nat)", ("actor/distillation/student_entropy", "actor/distillation/teacher_entropy", "actor/entropy")),
    ("Mixing weights", ("actor/distillation/forward_weight", "actor/distillation/jsd_beta")),
    ("Top-k probability mass", ("actor/distillation/student_mass", "actor/distillation/teacher_mass", "actor/distillation/teacher_topk_coverage")),
    ("Response length", ("response_length/mean", "response_length/max")),
    ("Prompt length", ("prompt_length/mean", "prompt_length/max")),
    ("Batch-width match and aborted ratio (not true truncation)", ("response_length/clip_ratio", "prompt_length/clip_ratio", "response/aborted_ratio")),
    ("Gradient norm", ("actor/grad_norm",)),
    ("Policy-gradient clipping", ("actor/distillation/pg_clipfrac",)),
    ("Rollout consistency", ("rollout_corr/kl", "rollout_corr/k3_kl", "training/rollout_probs_diff_mean")),
    ("Step time (s)", ("timing_s/step", "timing_s/gen", "timing_s/update_actor")),
    ("GPU memory (GiB)", ("actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb")),
    ("Outcome score", ("critic/score/mean", "critic/rewards/mean")),
)
COLORS = ("#2563eb", "#ea580c", "#16a34a", "#9333ea")
PADDING_MEAN_KEYS = tuple(
    f"actor/distillation/{name}" for name in (
        "student_mass", "teacher_mass", "student_entropy", "teacher_entropy",
        "teacher_topk_coverage", "jsd_beta", "forward_weight",
        "student_tail_mass", "teacher_tail_mass", "entropy_gate_confidence",
    )
)


def exclude_padding_means(rows, steps):
    """Mask explicitly reviewed suspect diagnostics, without estimating replacements."""
    return [
        {key: value for key, value in row.items()
         if row["step"] not in steps or key not in PADDING_MEAN_KEYS}
        for row in rows
    ]


def detect_padding_mean_dips(rows):
    """Conservatively flag isolated common-scale dips, not arbitrary outliers.

    Legacy logs lack valid-token/microbatch counts, so this is a heuristic,
    never a reconstruction of the true mean. Require both adjacent steps to
    recover, stable surrounding masses, and agreement across three diagnostics.
    Sustained distribution changes and loss/gradient/length spikes are untouched.
    """
    keys = [
        "actor/distillation/teacher_mass", "actor/distillation/student_mass",
        "actor/distillation/teacher_topk_coverage",
    ]
    findings = []
    for before, row, after in zip(rows, rows[1:], rows[2:]):
        if not (before["step"] + 1 == row["step"] == after["step"] - 1):
            continue
        if not all(key in item for item in (before, row, after) for key in keys):
            continue
        ratios = []
        for key in keys:
            left, value, right = before[key], row[key], after[key]
            if min(left, right) < 0.75 or abs(left - right) > 0.035:
                break
            if min(left, right) - value < 0.045:
                break
            ratios.append(value / ((left + right) / 2))
        if len(ratios) != len(keys):
            continue
        factor = statistics.median(ratios)
        if not (0.85 <= factor <= 0.95) or max(ratios) - min(ratios) > 0.025:
            continue
        findings.append({
            "step": row["step"],
            "reason": "suspected_padding_common_scale_dip",
            "estimated_common_scale": factor,
            "diagnostic_ratios_to_neighbors": dict(zip(keys, ratios)),
        })
    return findings


def read_metrics(path):
    """Keep the last record per step, allowing resumed runs and partial final lines."""
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if line_number == 1 or line.endswith("\n"):
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from None
                break  # interrupted write at end of a running job
            if not isinstance(item, dict) or not isinstance(item.get("data"), dict):
                continue
            step = item.get("step")
            if isinstance(step, bool) or not isinstance(step, int):
                continue
            row = {"step": step}
            for key, value in item["data"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    row[key] = float(value)
            rows[step] = row
    return [rows[step] for step in sorted(rows)]


def render_svg(rows, excluded_steps=()):
    width, height = 1500, 80 + math.ceil(len(PANELS) / 2) * 240 + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1e293b}.title{font-size:22px;font-weight:bold}.panel{font-size:16px;font-weight:bold}.axis{font-size:11px;fill:#64748b}.legend{font-size:10px}</style>',
        '<text x="30" y="39" class="title">Training metrics</text>',
        f'<text x="30" y="59" class="axis">{len(rows)} steps; last step {rows[-1]["step"]}</text>',
    ]
    if excluded_steps:
        note = ("Gaps: suspected padding-biased diagnostic means omitted at steps "
                + ", ".join(map(str, sorted(excluded_steps)))
                + "; no interpolation; loss/length unchanged.")
        parts.append(f'<text x="30" y="75" class="axis">{escape(note)}</text>')
    panel_w, panel_h = 710, 225
    for index, (title, keys) in enumerate(PANELS):
        px = 30 + (index % 2) * 740
        py = 80 + (index // 2) * 240
        parts.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" rx="10" fill="white" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{px+16}" y="{py+25}" class="panel">{escape(title)}</text>')
        left, right, top, bottom = px + 60, px + panel_w - 15, py + 65, py + panel_h - 24
        series = [(key, [(r["step"], r[key]) for r in rows if key in r]) for key in keys]
        series = [(key, values) for key, values in series if values]
        if not series:
            parts.append(f'<text x="{left}" y="{top+32}" class="axis">No data</text>')
            continue
        values = [value for _, points in series for _, value in points]
        low, high = min(values), max(values)
        if low == high:
            pad = max(abs(low) * 0.1, 0.01)
            low -= pad
            high += pad
        else:
            pad = (high - low) * 0.08
            low -= pad
            high += pad
        min_step, max_step = rows[0]["step"], rows[-1]["step"]
        step_span = max(1, max_step - min_step)
        parts.append(f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="#94a3b8"/>')
        for fraction in (0, 0.5, 1):
            y = bottom - fraction * (bottom - top)
            label = low + fraction * (high - low)
            parts.append(f'<text x="{left-7}" y="{y+4:.1f}" text-anchor="end" class="axis">{label:.3g}</text>')
        parts.append(f'<text x="{left}" y="{bottom+17}" class="axis">{min_step}</text>')
        parts.append(f'<text x="{right}" y="{bottom+17}" text-anchor="end" class="axis">{max_step}</text>')
        for number, (key, points) in enumerate(series):
            color = COLORS[number % len(COLORS)]
            segments = [[]]
            for row in rows:
                if key not in row:
                    if segments[-1]:
                        segments.append([])
                    continue
                segments[-1].append((
                    left + (row["step"] - min_step) / step_span * (right - left),
                    bottom - (row[key] - low) / (high - low) * (bottom - top),
                ))
            for xy in segments:
                if len(xy) == 1:
                    x, y = xy[0]
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
                elif xy:
                    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
                    parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8"/>')
            label = key.removeprefix("actor/distillation/")
            lx = px + 16 + (number % 2) * 340
            ly = py + 43 + (number // 2) * 12
            parts.append(f'<text x="{lx}" y="{ly}" class="legend" style="fill:{color}">● {escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="veRL file logger JSONL")
    parser.add_argument("--output", type=Path, help="SVG path; defaults beside metrics JSONL")
    parser.add_argument(
        "--exclude-padding-steps", type=int, nargs="+", default=[],
        help="Explicitly reviewed suspect steps: omit only distillation diagnostic means, leaving gaps.",
    )
    parser.add_argument(
        "--padding-filter", choices=("auto", "off"), default="auto",
        help="auto (default): conservatively mask isolated common-scale diagnostic dips; "
             "off: show original values. This heuristic cannot prove padding contamination.",
    )
    args = parser.parse_args()
    rows = read_metrics(args.metrics)
    if not rows:
        print(f"No training metrics yet in {args.metrics}")
        return
    findings = detect_padding_mean_dips(rows) if args.padding_filter == "auto" else []
    excluded_steps = set(args.exclude_padding_steps) | {item["step"] for item in findings}
    unknown_steps = excluded_steps - {row["step"] for row in rows}
    if unknown_steps:
        parser.error(f"Excluded steps absent from metrics: {sorted(unknown_steps)}")
    raw_rows = rows
    rows = exclude_padding_means(raw_rows, excluded_steps)
    output = args.output or args.metrics.with_name("training_curves.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(rows, excluded_steps), encoding="utf-8")
    selected_keys = list(dict.fromkeys(key for _, keys in PANELS for key in keys))
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", *selected_keys], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    audit_path = output.with_suffix(".filter.json")
    audit_path.write_text(json.dumps({
        "mode": args.padding_filter,
        "method": "isolated_common_scale_dip_v1",
        "caveat": "Heuristic suspects only; omitted values are not corrected estimates. "
                  "CSV blanks and SVG gaps indicate exclusions; original JSONL is unchanged.",
        "automatic_findings": findings,
        "manual_steps": sorted(set(args.exclude_padding_steps)),
        "excluded_steps": sorted(excluded_steps),
        "omitted_values": [
            {"step": row["step"], "values": {
                key: row[key] for key in PADDING_MEAN_KEYS if key in row
            }} for row in raw_rows if row["step"] in excluded_steps
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} steps to {output} and {csv_path}")
    print(f"Padding diagnostic filter: {sorted(excluded_steps)}; audit: {audit_path}")


if __name__ == "__main__":
    main()
