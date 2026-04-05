import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


METHOD_LABELS = {
    "eu": "EUPMU",
    "omd_tch": "OMD-TCH",
    "omd_tch_pgd": "OMD-TCH-PGD",
}


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    return float(value)


def format_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def average(values: List[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def find_trial_dirs(root: Path) -> List[Path]:
    trial_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("trial_")]
    return sorted(trial_dirs)


def read_trial_table(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(rows: List[Dict[str, str]], output_path: Path, fieldnames: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], output_path: Path, fieldnames: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(fieldnames) + " |\n"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |\n"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |\n")
    output_path.write_text("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="aggregate-trial-tables",
        description="Average Table 1 style per-trial CSVs into a single CSV and Markdown table.",
    )
    parser.add_argument(
        "--evaluation_root",
        type=Path,
        required=True,
        help="Root folder containing trial_*/table_<method>.csv outputs.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="eu",
        help="Space-separated method ids to include.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        required=True,
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    methods = [method for method in args.methods.split() if method.strip()]
    trial_dirs = find_trial_dirs(args.evaluation_root)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial_* directories found under {args.evaluation_root}")

    fieldnames = ["Forget.Class"]
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        fieldnames.extend([f"{label} UA", f"{label} FID"])

    row_order: List[str] = []
    row_seen = set()
    per_method_values: Dict[str, Dict[str, Dict[str, List[Optional[float]]]]] = {}
    warnings: List[str] = []

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        ua_col = f"{label} UA"
        fid_col = f"{label} FID"
        per_method_values[method] = {}

        for trial_dir in trial_dirs:
            trial_csv = trial_dir / f"table_{method}.csv"
            if not trial_csv.exists():
                warnings.append(f"missing table for method={method} in {trial_dir}")
                continue

            rows = read_trial_table(trial_csv)
            for row in rows:
                forget_class = row["Forget.Class"]
                if forget_class not in row_seen:
                    row_seen.add(forget_class)
                    row_order.append(forget_class)

                class_store = per_method_values[method].setdefault(
                    forget_class,
                    {"ua": [], "fid": []},
                )
                class_store["ua"].append(parse_float(row.get(ua_col, "")))
                class_store["fid"].append(parse_float(row.get(fid_col, "")))

    output_rows: List[Dict[str, str]] = []
    for forget_class in row_order:
        row = {"Forget.Class": forget_class}
        for method in methods:
            label = METHOD_LABELS.get(method, method)
            values = per_method_values[method].get(forget_class, {"ua": [], "fid": []})
            row[f"{label} UA"] = format_value(average(values["ua"]))
            row[f"{label} FID"] = format_value(average(values["fid"]))
        output_rows.append(row)

    write_csv(output_rows, args.output_csv, fieldnames)
    write_markdown(output_rows, args.output_md, fieldnames)

    print("=======================================")
    print(f"Found trial directories: {len(trial_dirs)}")
    print(f"Saved averaged CSV table to: {args.output_csv}")
    print(f"Saved averaged Markdown table to: {args.output_md}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("=======================================")
