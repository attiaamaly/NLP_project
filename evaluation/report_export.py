"""
SiftOps — University Report Exporter
======================================
Reads metrics.json + failures.csv + comparison.csv and produces a single
report_data.json that is structured for direct copy-paste into a LaTeX/Word
university report, plus pretty-printed console tables.

Usage:
    python evaluation/report_export.py
    python evaluation/report_export.py --out evaluation/report_data.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

EVAL_DIR     = Path(__file__).resolve().parent
METRICS_JSON = EVAL_DIR / "metrics.json"
FAILURES_CSV = EVAL_DIR / "failures.csv"
COMPARE_CSV  = EVAL_DIR / "comparison.csv"

console = Console()


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_metrics() -> dict:
    if not METRICS_JSON.exists():
        console.print(f"[red]metrics.json not found at {METRICS_JSON}[/red]")
        sys.exit(1)
    with open(METRICS_JSON) as f:
        return json.load(f)


def load_failures() -> pd.DataFrame | None:
    if not FAILURES_CSV.exists():
        return None
    return pd.read_csv(FAILURES_CSV)


def load_comparison() -> pd.DataFrame | None:
    if not COMPARE_CSV.exists():
        return None
    return pd.read_csv(COMPARE_CSV)


# ── Pretty printing ───────────────────────────────────────────────────────────
def print_overall(metrics: dict):
    ov = metrics.get("overall", {})
    t = Table(title="Overall Metrics", show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float) and v <= 1.0:
            return f"{v:.1%}"
        return str(v)

    for k, v in ov.items():
        t.add_row(k.replace("_", " ").title(), fmt(v))
    console.print(t)


def print_per_category(metrics: dict):
    pc = metrics.get("per_category", {})
    if not pc:
        return
    t = Table(title="Per-Category Retrieval", header_style="bold cyan")
    t.add_column("Category", style="bold")
    t.add_column("Count", justify="right")
    t.add_column("Hit@1",  justify="right")
    t.add_column("Hit@5",  justify="right")
    t.add_column("Src Match", justify="right")

    for cat, vals in sorted(pc.items()):
        def fmt(v): return f"{v:.1%}" if isinstance(v, float) else (str(v) if v is not None else "N/A")
        t.add_row(
            cat,
            str(vals.get("count", 0)),
            fmt(vals.get("hit_at_1")),
            fmt(vals.get("hit_at_5")),
            fmt(vals.get("source_match_rate")),
        )
    console.print(t)


def print_failures(failures: pd.DataFrame):
    if failures is None or failures.empty:
        console.print("[green]No failures to display.[/green]")
        return
    top = failures.head(10)
    t = Table(title=f"Top Failures (showing {len(top)} of {len(failures)})", header_style="bold red")
    t.add_column("QID")
    t.add_column("Question", max_width=40)
    t.add_column("Expected", max_width=25)
    t.add_column("Retrieved", max_width=25)
    t.add_column("Error Category")

    for _, row in top.iterrows():
        t.add_row(
            str(row.get("question_id", "")),
            str(row.get("question", ""))[:40],
            str(row.get("expected_document", ""))[:25],
            str(row.get("retrieved_document", ""))[:25],
            str(row.get("error_category", "")),
        )
    console.print(t)


def print_comparison(comp: pd.DataFrame, metrics: dict):
    if comp is None or comp.empty:
        return
    rc = metrics.get("retrieval_comparison", {})
    t = Table(title="BM25 vs Dense Retrieval", header_style="bold magenta")
    t.add_column("System")
    t.add_column("Hit@1", justify="right")
    t.add_column("Hit@5", justify="right")

    t.add_row("BM25",  f"{rc.get('bm25_hit_at_1', 0):.1%}",  f"{rc.get('bm25_hit_at_5', 0):.1%}")
    t.add_row("Dense", f"{rc.get('dense_hit_at_1', 0):.1%}", f"{rc.get('dense_hit_at_5', 0):.1%}")
    console.print(t)


# ── Report builder ────────────────────────────────────────────────────────────
def build_report(metrics: dict, failures_df, comparison_df) -> dict:
    """
    Return a structured dict suitable for a university report.
    All floats are pre-formatted as percentages where appropriate.
    """
    ov = metrics.get("overall", {})

    def pct(v): return f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else (str(v) if v is not None else "N/A")

    report = {
        "section_1_overall_metrics": {
            "description": "Aggregate evaluation results across all 35 benchmark questions.",
            "total_questions":        ov.get("total_questions"),
            "in_scope_questions":     ov.get("in_scope_questions"),
            "out_of_scope_questions": ov.get("out_of_scope_questions"),
            "search_success_rate":    pct(ov.get("search_success_rate")),
            "chat_success_rate":      pct(ov.get("chat_success_rate")),
            "hit_at_1":               pct(ov.get("hit_at_1")),
            "hit_at_5":               pct(ov.get("hit_at_5")),
            "refusal_accuracy":       pct(ov.get("refusal_accuracy")),
            "source_match_rate":      pct(ov.get("source_match_rate")),
        },
        "section_2_per_category_metrics": {
            "description": "Retrieval performance broken down by document category.",
            "categories": {
                cat: {
                    "count":            vals.get("count"),
                    "hit_at_1":         pct(vals.get("hit_at_1")),
                    "hit_at_5":         pct(vals.get("hit_at_5")),
                    "source_match_rate": pct(vals.get("source_match_rate")),
                }
                for cat, vals in metrics.get("per_category", {}).items()
            },
        },
        "section_3_top_failures": {
            "description": "Questions where retrieval or generation failed, with classified error types.",
            "failures": [],
        },
        "section_4_retrieval_comparison": {
            "description": "BM25 keyword baseline vs FastEmbed dense retrieval comparison.",
        },
        "section_5_error_distribution": {
            "description": "Distribution of failure types across all evaluated questions.",
            "distribution": metrics.get("error_distribution", {}),
        },
    }

    # Failures
    if failures_df is not None and not failures_df.empty:
        top_failures = failures_df.head(15).to_dict(orient="records")
        report["section_3_top_failures"]["failures"] = top_failures
        report["section_3_top_failures"]["total_failures"] = len(failures_df)

    # Comparison
    rc = metrics.get("retrieval_comparison", {})
    if rc:
        report["section_4_retrieval_comparison"].update({
            "bm25_hit_at_1":  pct(rc.get("bm25_hit_at_1")),
            "dense_hit_at_1": pct(rc.get("dense_hit_at_1")),
            "bm25_hit_at_5":  pct(rc.get("bm25_hit_at_5")),
            "dense_hit_at_5": pct(rc.get("dense_hit_at_5")),
            "dense_wins":     rc.get("dense_wins"),
            "bm25_wins":      rc.get("bm25_wins"),
            "both_win":       rc.get("both_win"),
            "neither_wins":   rc.get("neither_wins"),
        })
    else:
        report["section_4_retrieval_comparison"]["note"] = (
            "Run with --compare flag to include BM25 vs Dense data."
        )

    return report


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export evaluation results for university report")
    parser.add_argument("--out", default=str(EVAL_DIR / "report_data.json"),
                        help="Output path for report JSON")
    args = parser.parse_args()

    metrics     = load_metrics()
    failures_df = load_failures()
    compare_df  = load_comparison()

    # Print to console
    console.rule("[bold blue]SiftOps — Evaluation Report")
    print_overall(metrics)
    print_per_category(metrics)
    print_failures(failures_df)
    print_comparison(compare_df, metrics)

    # Build + save JSON
    report = build_report(metrics, failures_df, compare_df)
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"\n[bold green]Report data saved → {out_path}[/bold green]")
    console.print("You can directly reference this file in your LaTeX or Word report.\n")


if __name__ == "__main__":
    main()