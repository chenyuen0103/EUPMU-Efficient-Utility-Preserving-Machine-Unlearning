#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SEED_ROOT_OVERRIDES = {
    # No seed overrides needed; all seeds collected from default root
}


DISPLAY_NAMES = {
    "retrain": "Retrain",
    "FT": "FT",
    "GA": "GA",
    "wfisher": "IU",
    "famo": "FAMO",
    "igs": "UNGrad",
    "FT_prune": "l1-sparse",
    "RL": "RL",
    "eu": "EUPMU",
    "eu_fast": "EUPMU-fast",
    "gdr_gma": "GDR-GMA",
    "chebyshev": "Chebyshev",
    "omd_tch_eg": "OMD-TCH-EG",
    "omd_tch_pgd": "OMD-TCH-PGD",
    "ada_omd_tch_eg": "AdaAFLeg",
    "RL_proximal": "SalUn-soft",
}

DESCRIPTION_NAMES = {
    "retrain": "Retrain: Full retraining after removing the forget set.",
    "RL": "RL: Random-label unlearning baseline on the forget set.",
    "FT": "FT: Plain fine-tuning baseline without special unlearning machinery.",
    "GA": "GA: Gradient-ascent baseline that pushes against the forget objective.",
    "wfisher": "IU: Weighted-Fisher influence-based unlearning baseline.",
    "famo": "FAMO: Efficient MGDA-style multi-objective weighting baseline.",
    "igs": "UNGrad: Explicit unilateral gradient-surgery baseline.",
    "FT_prune": "l1-sparse: Sparse fine-tuning / pruning-based unlearning baseline.",
    "gdr_gma": "GDR-GMA: Gradient surgery with direction rectification and magnitude adjustment.",
    "eu": "EUPMU: Efficient implicit unilateral gradient surgery.",
    "eu_fast": "EUPMU-fast: Faster approximation to EUPMU without extra retain recomputation.",
    "chebyshev": "Chebyshev: Augmented Tchebycheff scalarization baseline for retain/forget MOO.",
    "omd_tch_eg": "OMD-TCH-EG: Exponentiated-gradient online mirror-descent Tchebycheff method.",
    "omd_tch_pgd": "OMD-TCH-PGD: Projected-gradient online mirror-descent Tchebycheff method.",
    "ada_omd_tch_eg": "AdaAFLeg: Adaptive exponentiated-gradient OMD-TCH variant with marked-model aggregation.",
    "RL_proximal": "SalUn-soft: Soft-thresholding SalUn-style proximal baseline.",
}

CANONICAL_METHOD_IDS = {
    "omd_tch": "omd_tch_eg",
    "afleg": "omd_tch_eg",
    "afl": "omd_tch_pgd",
    "ada_afleg": "ada_omd_tch_eg",
}

METHOD_GROUPS = {
    "retrain": "retrain",
    "famo": "gradient_surgery",
    "igs": "gradient_surgery",
    "gdr_gma": "gradient_surgery",
    "eu": "gradient_surgery",
    "eu_fast": "gradient_surgery",
    "chebyshev": "tch",
    "omd_tch_eg": "tch",
    "omd_tch_pgd": "tch",
    "ada_omd_tch_eg": "tch",
}

GROUP_ORDER = {
    "retrain": 0,
    "other": 1,
    "gradient_surgery": 2,
    "tch": 3,
}

GROUP_METHOD_ORDER = {
    "retrain": 0,
    "RL": 0,
    "FT": 1,
    "GA": 2,
    "wfisher": 3,
    "FT_prune": 4,
    "famo": 0,
    "igs": 1,
    "gdr_gma": 2,
    "eu": 3,
    "eu_fast": 4,
    "chebyshev": 0,
    "omd_tch_eg": 1,
    "omd_tch_pgd": 2,
    "ada_omd_tch_eg": 3,
}


def canonical_method_id(method_id: str) -> str:
    if "/" not in method_id:
        return CANONICAL_METHOD_IDS.get(method_id, method_id)
    base, rest = method_id.split("/", 1)
    base = CANONICAL_METHOD_IDS.get(base, base)
    return f"{base}/{rest}"


def method_group(method_id: str) -> str:
    base_method = method_id.split("/")[0]
    return METHOD_GROUPS.get(base_method, "other")





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan classification experiment outputs and emit a LaTeX table in the style of Table 4."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/resnet18/cifar10/forget_10.0%"),
        help="Root directory that contains method subdirectories such as retrain/ and RL/.",
    )
    parser.add_argument(
        "--accuracy-key",
        default="accuracy",
        choices=["accuracy", "avg_accuracy", "adaptive_accuracy"],
        help=(
            "Which accuracy block to use from evaluation_result.json. "
            "Use avg_accuracy for OMD-TCH averaged-iterate results or adaptive_accuracy for AdaOMD-TCH marked-model results."
        ),
    )
    parser.add_argument(
        "--mia-key",
        default="confidence",
        choices=["correctness", "confidence", "entropy", "m_entropy", "prob"],
        help="Which SVC_MIA_forget_efficacy submetric to use for the MIA column.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional whitelist of method ids to include, e.g. retrain eu chebyshev omd_tch_eg. "
            "Method ids are inferred from directory names."
        ),
    )
    parser.add_argument(
        "--exclude-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional seed ids to exclude, e.g. --exclude-seeds 3.",
    )
    parser.add_argument(
        "--train-run",
        type=int,
        default=1,
        help=(
            "Only include rows whose folder name ends with this training-run index, e.g. 1 for seed_3_train_1. "
            "Set a different value to average over another train run."
        ),
    )
    parser.add_argument(
        "--only-train-tagged",
        action="store_true",
        help="Only include results under seed_*_train_* folders.",
    )
    parser.add_argument(
        "--match-seed-train",
        action="store_true",
        help="Only include rows where seed id equals train-run id (seed_k_train_k).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the LaTeX table. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--caption",
        default="Table 4 style results on CIFAR-10 class-wise forgetting (10\%).",
        help="Base LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:table4_like",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals to print.",
    )
    parser.add_argument(
        "--no-descriptions-in-caption",
        action="store_true",
        help="Do not append one-line method definitions to the caption.",
    )
    parser.add_argument(
        "--required-seeds",
        type=int,
        default=5,
        help="Require this many seeds per method before building the summary table.",
    )
    parser.add_argument(
        "--allow-partial-seeds",
        action="store_true",
        help="Allow summarizing methods even if they have fewer than --required-seeds runs.",
    )
    return parser.parse_args()


def format_stat(stat: Optional[Dict[str, float]], decimals: int) -> str:
    if stat is None:
        return "--"
    mean = stat["mean"]
    std = stat["std"]
    return f"{mean:.{decimals}f} {{\\scriptsize$\\pm$ {std:.{decimals}f}}}"


def format_ranked_stat(
    stat: Optional[Dict[str, float]],
    decimals: int,
    rank: Optional[int],
) -> str:
    if stat is None:
        return "--"
    mean_text = f"{stat['mean']:.{decimals}f}"
    std_text = f"{{\\scriptsize$\\pm$ {stat['std']:.{decimals}f}}}"
    if rank == 1:
        mean_text = f"\\textbf{{{mean_text}}}"
    elif rank == 2:
        mean_text = f"\\underline{{{mean_text}}}"
    return f"{mean_text} {std_text}"


def infer_method_id(json_path: Path, root: Path) -> str:
    rel = json_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 2:
        return parts[0] if parts else json_path.stem

    top = parts[0]
    if top == "RL":
        if len(parts) >= 4 and parts[2].startswith("seed_"):
            return parts[1]
        if len(parts) >= 3 and parts[1].startswith("seed_"):
            return "RL"
        if len(parts) >= 4 and parts[1] != "None" and parts[2] != "None":
            return f"{parts[1]}/{parts[2]}"
        if len(parts) >= 3 and parts[1] != "None":
            return parts[1]

    # When the root is already at .../RL, sweep runs look like:
    #   omd_tch_eg/seed_5_train_1/eta_.../evaluation_result.json
    # Group those by the hyperparameter-setting folder so each variant gets
    # one average across seeds.
    if len(parts) >= 3 and parts[1].startswith("seed_"):
        if len(parts) >= 4 and not parts[2].endswith(".json"):
            return f"{parts[0]}/{parts[2]}"
        return parts[0]
    return top


def infer_seed_id(json_path: Path, root: Path) -> Optional[int]:
    rel = json_path.relative_to(root)
    for part in rel.parts:
        if part.startswith("seed_"):
            try:
                # Extract seed number from format like 'seed_1', 'seed_1_train_1', etc.
                seed_part = part.split("_", 1)[1]
                # If format is 'seed_N_train_M', extract N
                if "_train_" in seed_part:
                    seed_part = seed_part.split("_train_", 1)[0]
                return int(seed_part)
            except ValueError:
                return None
    return None


def infer_train_run_id(json_path: Path, root: Path) -> Optional[int]:
    rel = json_path.relative_to(root)
    for part in rel.parts:
        if "_train_" not in part:
            continue
        try:
            return int(part.rsplit("_train_", 1)[1])
        except ValueError:
            return None
    return None




def format_display_name(method_id: str) -> str:
    if "/" not in method_id:
        return DISPLAY_NAMES.get(method_id, method_id)
    base, tag = method_id.split("/", 1)
    base_name = DISPLAY_NAMES.get(base, base)
    if tag.startswith("eta_") and tag.count("_") == 1:
        eta = tag.split("eta_", 1)[1].replace("p", ".")
        return f"{base_name} ({eta})"
    return f"{base_name} [{tag}]"


def load_row(json_path: Path, root: Path, accuracy_key: str, mia_key: str) -> Optional[Dict[str, object]]:
    data = json.loads(json_path.read_text())
    accuracy = data.get(accuracy_key)
    if not isinstance(accuracy, dict):
        return None

    mia = data.get("SVC_MIA_forget_efficacy", {})
    if not isinstance(mia, dict):
        mia = {}

    ua = accuracy.get("forget")
    ra = accuracy.get("retain")
    ta = accuracy.get("test")
    mia_value = mia.get(mia_key)

    numeric_values = [ua, ra, ta, mia_value]
    avg_score = None
    if all(isinstance(v, (int, float)) for v in numeric_values):
        avg_score = sum(float(v) for v in numeric_values) / 4.0

    method_id = canonical_method_id(infer_method_id(json_path, root))
    return {
        "method_id": method_id,
        "display_name": format_display_name(method_id),
        "json_path": json_path,
        "seed_id": infer_seed_id(json_path, root),
        "train_run_id": infer_train_run_id(json_path, root),
        "ua": ua,
        "ra": ra,
        "ta": ta,
        "mia": mia_value,
        "avg_score": avg_score,
    }


def collect_rows(
    root: Path,
    accuracy_key: str,
    mia_key: str,
    train_run: Optional[int],
    only_train_tagged: bool,
    match_seed_train: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for json_path in sorted(root.glob("**/evaluation_result.json")):
        row = load_row(json_path, root, accuracy_key, mia_key)
        if row is not None:
            if only_train_tagged and row.get("train_run_id") is None:
                continue
            effective_train_run_id = row.get("train_run_id")
            # Retrain folders are typically seed_k (no _train_ tag); treat them as train_run=1.
            if (
                effective_train_run_id is None
                and row.get("method_id") == "retrain"
                and isinstance(row.get("seed_id"), int)
            ):
                effective_train_run_id = 1
            if train_run is not None and train_run >= 0 and effective_train_run_id != train_run:
                continue
            if match_seed_train and row.get("seed_id") != row.get("train_run_id"):
                continue
            rows.append(row)
    return rows


def summarize_metric(rows: List[Dict[str, object]], key: str) -> Optional[Dict[str, float]]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(values)}


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        method_id = canonical_method_id(str(row["method_id"]))
        row["method_id"] = method_id
        row["display_name"] = format_display_name(method_id)
        grouped.setdefault(method_id, []).append(row)

    aggregated: List[Dict[str, object]] = []
    for method_id, method_rows in grouped.items():
        aggregated.append(
            {
                "method_id": method_id,
                "display_name": format_display_name(method_id),
                "json_paths": [row["json_path"] for row in method_rows],
                "n": len(method_rows),
                "ua": summarize_metric(method_rows, "ua"),
                "ra": summarize_metric(method_rows, "ra"),
                "ta": summarize_metric(method_rows, "ta"),
                "mia": summarize_metric(method_rows, "mia"),
                "avg_score": summarize_metric(method_rows, "avg_score"),
            }
        )
    return aggregated


def validate_seed_counts(
    rows: List[Dict[str, object]],
    required_seeds: int,
) -> List[str]:
    incomplete = []
    for row in sorted(rows, key=method_sort_key):
        n = int(row.get("n", 0))
        if n < required_seeds:
            incomplete.append(f"{row['display_name']}: {n}/{required_seeds}")
    return incomplete


def method_sort_key(row: Dict[str, object]) -> Tuple[int, str]:
    method_id = str(row["method_id"])
    base_method = method_id.split('/')[0]
    group = method_group(method_id)
    return (
        GROUP_ORDER.get(group, 999),
        GROUP_METHOD_ORDER.get(base_method, 999),
        str(row["display_name"]),
    )


def compute_column_ranks(
    rows: List[Dict[str, object]],
    key: str,
    exclude_method_ids: Optional[List[str]] = None,
) -> Dict[str, Optional[int]]:
    excluded = set(exclude_method_ids or [])
    values = []
    for row in rows:
        method_id = str(row["method_id"])
        if method_id in excluded:
            continue
        stat = row.get(key)
        if isinstance(stat, dict) and isinstance(stat.get("mean"), (int, float)):
            values.append(float(stat["mean"]))

    unique_values = sorted(set(values), reverse=True)
    top_values = unique_values[:2]

    ranks: Dict[str, Optional[int]] = {}
    for row in rows:
        method_id = str(row["method_id"])
        if method_id in excluded:
            ranks[method_id] = None
            continue
        stat = row.get(key)
        if not isinstance(stat, dict) or not isinstance(stat.get("mean"), (int, float)):
            ranks[method_id] = None
            continue
        mean_value = float(stat["mean"])
        if top_values and mean_value == top_values[0]:
            ranks[method_id] = 1
        elif len(top_values) > 1 and mean_value == top_values[1]:
            ranks[method_id] = 2
        else:
            ranks[method_id] = None
    return ranks


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def build_caption(base_caption: str, rows: List[Dict[str, object]], include_descriptions: bool) -> str:
    base_caption = base_caption + " Results are reported as mean \\pm std over available seeds."
    if not include_descriptions:
        return base_caption
    description_parts = []
    seen = set()
    for row in sorted(rows, key=method_sort_key):
        method_id = str(row["method_id"])
        if method_id in seen:
            continue
        seen.add(method_id)
        if method_id in DESCRIPTION_NAMES:
            description_parts.append(DESCRIPTION_NAMES[method_id])
    if not description_parts:
        return base_caption
    return base_caption + " " + " ".join(description_parts)


def build_table(
    rows: List[Dict[str, object]],
    caption: str,
    label: str,
    decimals: int,
    mia_key: str,
    accuracy_key: str,
    include_descriptions: bool,
) -> str:
    full_caption = latex_escape(build_caption(caption, rows, include_descriptions))
    avg_ranks = compute_column_ranks(rows, "avg_score", exclude_method_ids=["retrain"])
    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        f"\\caption{{{full_caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        f"Method & UA & RA & TA & MIA ({mia_key}) & Avg. score " + r"\\",
        r"\midrule",
    ]

    body = []
    previous_group: Optional[str] = None
    for row in sorted(rows, key=method_sort_key):
        method_id = str(row["method_id"])
        current_group = method_group(method_id)
        if previous_group is not None and current_group != previous_group:
            body.append(r"\hline")
        line = (
            f"{row['display_name']} & "
            f"{format_stat(row.get('ua'), decimals)} & "
            f"{format_stat(row.get('ra'), decimals)} & "
            f"{format_stat(row.get('ta'), decimals)} & "
            f"{format_stat(row.get('mia'), decimals)} & "
            f"{format_ranked_stat(row.get('avg_score'), decimals, avg_ranks.get(method_id))} " + r"\\"
        )
        body.append(line)
        previous_group = current_group

    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        f"% accuracy_key={accuracy_key}, mia_key={mia_key}",
        r"\end{table}",
    ]
    return "\n".join(header + body + footer) + "\n"


def report_seed_counts(rows: List[Dict[str, object]]) -> None:
    print("Seed counts used for averages:")
    for row in sorted(rows, key=method_sort_key):
        metric_counts = []
        for label, key in [
            ("UA", "ua"),
            ("RA", "ra"),
            ("TA", "ta"),
            ("MIA", "mia"),
            ("Avg", "avg_score"),
        ]:
            stat = row.get(key)
            n = stat.get("n", 0) if isinstance(stat, dict) else 0
            metric_counts.append(f"{label}={n}")
        print(f"- {row['display_name']}: " + ", ".join(metric_counts))
    print()


def main() -> None:
    args = parse_args()
    rows = collect_rows(
        args.root,
        args.accuracy_key,
        args.mia_key,
        args.train_run,
        args.only_train_tagged,
        args.match_seed_train,
    )

    # Allow seed-specific result roots, e.g. using rerun artifacts for one seed.
    # for seed_id, override_root in SEED_ROOT_OVERRIDES.items():
    #     if not override_root.exists():
    #         continue
    #     rows = [row for row in rows if row.get("seed_id") != seed_id]
    #     rows.extend(
    #         collect_rows(
    #             override_root,
    #             args.accuracy_key,
    #             args.mia_key,
    #             args.train_run,
    #             args.only_train_tagged,
    #             args.match_seed_train,
    #         )
    #     )

    if args.exclude_seeds:
        excluded = set(args.exclude_seeds)
        rows = [row for row in rows if row.get("seed_id") not in excluded]

    if args.methods is not None:
        allowed = {canonical_method_id(method_id) for method_id in args.methods}
        rows = [
            row
            for row in rows
            if canonical_method_id(str(row["method_id"])) in allowed
            or canonical_method_id(str(row["method_id"])) .split("/", 1)[0] in allowed
        ]
    else:
        rows = [row for row in rows if "/" not in str(row["method_id"])]

    rows = aggregate_rows(rows)

    if not rows:
        raise SystemExit(f"No usable evaluation_result.json files found under {args.root}")

    if not args.allow_partial_seeds:
        incomplete = validate_seed_counts(rows, args.required_seeds)
        if incomplete:
            raise SystemExit(
                "Incomplete seed coverage. Methods below do not have the required number of seeds:\n"
                + "\n".join(incomplete)
            )

    report_seed_counts(rows)

    table = build_table(
        rows,
        args.caption,
        args.label,
        args.decimals,
        args.mia_key,
        args.accuracy_key,
        include_descriptions=not args.no_descriptions_in_caption,
    )

    if args.output is not None:
        args.output.write_text(table)
        print(f"Saved LaTeX table to {args.output}")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
