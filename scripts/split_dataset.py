#split_dataset.py

import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "dataset.jsonl"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "data" / "reports"

OUTPUT_FILES = {
    "train": SPLIT_DIR / "train.jsonl",
    "validation": SPLIT_DIR / "validation.jsonl",
    "test": SPLIT_DIR / "test.jsonl",
}

REPORT_FILE = REPORT_DIR / "split_report.json"

SEED = "arabic-gulf-ai-v04"

SPLIT_WEIGHTS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


def stable_hash(value):
    return hashlib.sha256(
        f"{SEED}:{value}".encode("utf-8")
    ).hexdigest()

def read_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                )

            normalized_text = record.get("normalized_text")

            if not isinstance(normalized_text, str):
                raise ValueError(
                    f"Missing normalized_text at line {line_number}"
                )

            normalized_text = normalized_text.strip()

            if not normalized_text:
                raise ValueError(
                    f"Empty normalized_text at line {line_number}"
                )

            record["_split_fingerprint"] = hashlib.sha256(
                normalized_text.encode("utf-8")
            ).hexdigest()

            record["_split_dialect"] = (
                record.get("dialect") or "unknown"
            )

            records.append(record)

    return records


def build_groups(records):
    """
    Records with identical normalized text are kept
    in the same group to prevent exact-text leakage.
    """

    groups = defaultdict(list)

    for record in records:
        fingerprint = record["_split_fingerprint"]
        groups[fingerprint].append(record)

    result = []

    for fingerprint, group_records in groups.items():

        dialect_counts = Counter(
            record["_split_dialect"]
            for record in group_records
        )

        result.append({
            "fingerprint": fingerprint,
            "records": group_records,
            "size": len(group_records),
            "dialects": dialect_counts,
            "sort_key": stable_hash(fingerprint),
        })

    return result


def calculate_targets(total):
    """
    Calculate target record counts using 80/10/10.
    When at least 3 unique groups exist, each split
    receives at least one group.
    """

    names = ["train", "validation", "test"]

    targets = {
        name: int(total * SPLIT_WEIGHTS[name])
        for name in names
    }

    remainders = {
        name: total * SPLIT_WEIGHTS[name] - targets[name]
        for name in names
    }

    remaining = total - sum(targets.values())

    for name in sorted(
        names,
        key=lambda item: remainders[item],
        reverse=True,
    ):
        if remaining <= 0:
            break

        targets[name] += 1
        remaining -= 1

    if total >= 3:
        for name in names:
            if targets[name] == 0:
                donor = max(
                    names,
                    key=lambda item: targets[item]
                )

                targets[donor] -= 1
                targets[name] += 1

    return targets


def calculate_objective(
    split_name,
    group,
    record_counts,
    dialect_counts,
    global_targets,
    dialect_targets,
):
    """
    Score a potential assignment.

    Lower scores indicate a better fit to the
    global and dialect-specific target distributions.
    """

    score = 0.0

    size = group["size"]

    for name in SPLIT_WEIGHTS:

        projected = record_counts[name]

        if name == split_name:
            projected += size

        target = global_targets[name]
        denominator = max(target, 1)

        score += (
            (projected - target) ** 2
        ) / denominator

    for dialect, dialect_size in group["dialects"].items():

        targets = dialect_targets[dialect]

        for name in SPLIT_WEIGHTS:

            projected = dialect_counts[dialect][name]

            if name == split_name:
                projected += dialect_size

            target = targets[name]
            denominator = max(target, 1)

            score += 2.0 * (
                (projected - target) ** 2
            ) / denominator

    return score


def assign_groups(groups):
    total_records = sum(group["size"] for group in groups)

    assignments = {
        "train": [],
        "validation": [],
        "test": [],
    }

    if not groups:
        return assignments

    # With fewer than three unique groups, keep
    # everything in training rather than creating
    # misleading validation/test datasets.
    if len(groups) < 3:
        assignments["train"] = groups
        return assignments

    global_targets = calculate_targets(total_records)

    dialect_totals = Counter()

    for group in groups:
        dialect_totals.update(group["dialects"])

    dialect_targets = {
        dialect: calculate_targets(count)
        for dialect, count in dialect_totals.items()
    }

    record_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    dialect_counts = defaultdict(
        lambda: {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
    )

    # Deterministic ordering.
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -group["size"],
            group["sort_key"],
        ),
    )

    # Reserve one unique group for every split.
    reserved = sorted(
        groups,
        key=lambda group: group["sort_key"]
    )

    split_names = ["train", "validation", "test"]

    for index, split_name in enumerate(split_names):

        group = reserved[index]

        assignments[split_name].append(group)
        record_counts[split_name] += group["size"]

        for dialect, count in group["dialects"].items():
            dialect_counts[dialect][split_name] += count

    reserved_fingerprints = {
        group["fingerprint"]
        for group in reserved[:3]
    }

    remaining_groups = [
        group for group in ordered_groups
        if group["fingerprint"] not in reserved_fingerprints
    ]

    # Greedily place remaining groups where they
    # best fit the target distribution.
    for group in remaining_groups:

        scores = {}

        for split_name in split_names:

            scores[split_name] = calculate_objective(
                split_name,
                group,
                record_counts,
                dialect_counts,
                global_targets,
                dialect_targets,
            )

        selected_split = min(
            split_names,
            key=lambda name: scores[name]
        )

        assignments[selected_split].append(group)

        record_counts[selected_split] += group["size"]

        for dialect, count in group["dialects"].items():
            dialect_counts[dialect][selected_split] += count

    return assignments


def clean_record(record):
    """
    Remove internal splitting fields before writing JSONL.
    """

    cleaned = dict(record)

    cleaned.pop("_split_fingerprint", None)
    cleaned.pop("_split_dialect", None)

    return cleaned


def write_splits(assignments):

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, output_file in OUTPUT_FILES.items():

        records = []

        for group in assignments[split_name]:
            records.extend(group["records"])

        with output_file.open("w", encoding="utf-8") as file:

            for record in records:
                file.write(
                    json.dumps(
                        clean_record(record),
                        ensure_ascii=False,
                    ) + "\n"
                )


def verify_no_leakage(assignments):

    fingerprint_locations = defaultdict(set)

    for split_name, groups in assignments.items():

        for group in groups:
            fingerprint_locations[
                group["fingerprint"]
            ].add(split_name)

    leaked = {
        fingerprint: sorted(locations)
        for fingerprint, locations in fingerprint_locations.items()
        if len(locations) > 1
    }

    return leaked


def create_report(records, groups, assignments, leaked):

    report = {
        "version": "0.4",
        "seed": SEED,
        "total_records": len(records),
        "unique_text_groups": len(groups),
        "duplicate_records": len(records) - len(groups),
        "exact_text_leakage_groups": len(leaked),
        "splits": {},
        "dialect_distribution": {},
        "status": "passed" if not leaked else "failed",
    }

    dialect_summary = defaultdict(
        lambda: {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
    )

    for split_name, split_groups in assignments.items():

        split_records = []

        for group in split_groups:
            split_records.extend(group["records"])

        dialect_counts = Counter(
            record["_split_dialect"]
            for record in split_records
        )

        report["splits"][split_name] = {
            "records": len(split_records),
            "unique_groups": len(split_groups),
            "percentage": round(
                len(split_records) / max(len(records), 1) * 100,
                2,
            ),
            "dialects": dict(dialect_counts),
        }

        for dialect, count in dialect_counts.items():
            dialect_summary[dialect][split_name] = count

    report["dialect_distribution"] = dict(dialect_summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return report


def main():

    print("\n===== ARABIC GULF AI DATA SPLITTING v0.4 =====\n")

    records = read_dataset()

    groups = build_groups(records)

    assignments = assign_groups(groups)

    leaked = verify_no_leakage(assignments)

    write_splits(assignments)

    report = create_report(
        records,
        groups,
        assignments,
        leaked,
    )

    print(f"Total records: {report['total_records']}")
    print(f"Unique text groups: {report['unique_text_groups']}")
    print(f"Duplicate records: {report['duplicate_records']}")

    print("\nSplit distribution:")

    for split_name, details in report["splits"].items():

        print(
            f"{split_name}: "
            f"{details['records']} records "
            f"({details['percentage']}%)"
        )

        print(f"  Dialects: {details['dialects']}")

    print("\nLeakage verification:")
    print(f"Exact-text leakage groups: {report['exact_text_leakage_groups']}")

    print(f"\nStatus: {report['status']}")

    print(f"\nReport saved to:\n{REPORT_FILE}")

    if leaked:
        raise RuntimeError(
            "Exact duplicate leakage detected between splits."
        )

    print("\nData splitting completed successfully.")


if __name__ == "__main__":
    main()
