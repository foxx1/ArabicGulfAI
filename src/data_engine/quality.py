
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_FILE = ROOT / "data" / "processed" / "dataset.jsonl"
REPORT_DIR = ROOT / "data" / "reports"
REPORT_FILE = REPORT_DIR / "quality_report.json"
INDEX_FILE = REPORT_DIR / ".quality_index.sqlite"

REQUIRED_FIELDS = [
    "id",
    "text",
    "normalized_text",
    "language",
    "dialect",
]

REQUIRED_METADATA = [
    "source",
    "license",
    "consent_status",
]

VALID_DIALECTS = {
    "bahraini",
    "saudi",
    "kuwaiti",
    "emirati",
    "qatari",
    "omani",
    "msa",
    "unknown",
}

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_PATTERN.search(text))



def create_index():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(INDEX_FILE)

    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -64000")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS ids (
            record_id TEXT PRIMARY KEY
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            fingerprint TEXT PRIMARY KEY
        )
    """)

    # Clear previous audit data instead of deleting the database file.
    connection.execute("DELETE FROM ids")
    connection.execute("DELETE FROM fingerprints")

    connection.commit()
    return connection


def validate_record(record, line_number, connection, report):
    errors = report["errors"]
    warnings = report["warnings"]

    if not isinstance(record, dict):
        errors.append({
            "line": line_number,
            "message": "Record must be a JSON object"
        })
        return False

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in record
    ]

    if missing_fields:
        errors.append({
            "line": line_number,
            "message": "Missing required fields",
            "fields": missing_fields
        })
        return False

    invalid_fields = [
        field for field in REQUIRED_FIELDS
        if not isinstance(record.get(field), str)
    ]

    if invalid_fields:
        errors.append({
            "line": line_number,
            "message": "Invalid field types",
            "fields": invalid_fields
        })
        return False

    record_id = record["id"].strip()
    text = record["text"].strip()
    normalized = record["normalized_text"].strip()
    dialect = record["dialect"].strip().lower()

    if not record_id:
        errors.append({
            "line": line_number,
            "message": "Empty record ID"
        })
        return False

    if not text or not normalized:
        errors.append({
            "line": line_number,
            "message": "Empty text or normalized text"
        })
        return False

    try:
        connection.execute(
            "INSERT INTO ids(record_id) VALUES (?)",
            (record_id,)
        )
    except sqlite3.IntegrityError:
        errors.append({
            "line": line_number,
            "message": "Duplicate record ID",
            "record_id": record_id
        })

    fingerprint = sha256_text(normalized)

    try:
        connection.execute(
            "INSERT INTO fingerprints(fingerprint) VALUES (?)",
            (fingerprint,)
        )
    except sqlite3.IntegrityError:
        warnings.append({
            "line": line_number,
            "message": "Duplicate normalized text",
            "record_id": record_id
        })

    if record["language"] != "ar":
        warnings.append({
            "line": line_number,
            "message": "Unexpected language value",
            "value": record["language"]
        })

    if dialect not in VALID_DIALECTS:
        warnings.append({
            "line": line_number,
            "message": "Unrecognized dialect",
            "value": dialect
        })

    if not contains_arabic(text):
        warnings.append({
            "line": line_number,
            "message": "No Arabic characters detected"
        })

    for field in REQUIRED_METADATA:
        value = record.get(field)

        if not isinstance(value, str) or not value.strip():
            warnings.append({
                "line": line_number,
                "message": "Missing metadata",
                "field": field
            })

    return True


def audit_dataset(data_file=DEFAULT_DATA_FILE):
    data_file = Path(data_file)

    report = {
        "project": "Arabic Gulf AI",
        "pipeline_version": "0.3",
        "dataset_file": str(data_file),
        "total_records": 0,
        "valid_records": 0,
        "arabic_records": 0,
        "duplicate_texts": 0,
        "average_characters": 0,
        "min_characters": 0,
        "max_characters": 0,
        "dialect_distribution": {},
        "domain_distribution": {},
        "errors": [],
        "warnings": [],
        "status": "not_started"
    }

    if not data_file.exists():
        report["errors"].append({
            "line": 0,
            "message": "Dataset file not found"
        })
        report["status"] = "failed"
        return report

    connection = create_index()

    dialect_counts = Counter()
    domain_counts = Counter()

    total_characters = 0
    min_characters = None
    max_characters = 0
    arabic_records = 0
    duplicate_texts = 0

    try:
        with data_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    report["warnings"].append({
                        "line": line_number,
                        "message": "Blank line"
                    })
                    continue

                report["total_records"] += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    report["errors"].append({
                        "line": line_number,
                        "message": "Invalid JSON",
                        "detail": str(exc)
                    })
                    continue

                valid = validate_record(
                    record,
                    line_number,
                    connection,
                    report
                )

                if not valid:
                    continue

                text = record["text"].strip()
                normalized = record["normalized_text"].strip()
                dialect = record["dialect"].strip().lower()
                domain = record.get("domain", "unspecified")

                report["valid_records"] += 1

                char_count = len(text)
                total_characters += char_count

                min_characters = (
                    char_count if min_characters is None
                    else min(min_characters, char_count)
                )

                max_characters = max(
                    max_characters,
                    char_count
                )

                if contains_arabic(text):
                    arabic_records += 1

                dialect_counts[dialect] += 1
                domain_counts[str(domain)] += 1

                fingerprint = sha256_text(normalized)

                cursor = connection.execute(
                    "SELECT 1 FROM fingerprints WHERE fingerprint = ?",
                    (fingerprint,)
                )

                # The uniqueness check is performed during insertion.
                # Duplicate warnings are counted from the warning list.
                # This variable is retained for report clarity.

        connection.commit()

        duplicate_texts = sum(
            1 for item in report["warnings"]
            if item["message"] == "Duplicate normalized text"
        )

        count = report["valid_records"]

        report["arabic_records"] = arabic_records
        report["duplicate_texts"] = duplicate_texts
        report["average_characters"] = round(
            total_characters / count, 2
        ) if count else 0

        report["min_characters"] = min_characters or 0
        report["max_characters"] = max_characters

        report["dialect_distribution"] = dict(dialect_counts)
        report["domain_distribution"] = dict(domain_counts)

        report["status"] = (
            "passed" if not report["errors"]
            else "failed"
        )

    finally:
        connection.close()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )

    return report


def print_report(report):
    print("\n===== ARABIC GULF AI QUALITY AUDIT v0.3 =====")

    print("Status:", report["status"])
    print("Total records:", report["total_records"])
    print("Valid records:", report["valid_records"])
    print("Arabic records:", report["arabic_records"])
    print("Duplicate texts:", report["duplicate_texts"])

    print("Average characters:", report["average_characters"])
    print("Minimum characters:", report["min_characters"])
    print("Maximum characters:", report["max_characters"])

    print("\nDialect distribution:")
    for name, count in sorted(
        report["dialect_distribution"].items()
    ):
        print(f"{name}: {count}")

    print("\nDomain distribution:")
    for name, count in sorted(
        report["domain_distribution"].items()
    ):
        print(f"{name}: {count}")

    print("\nErrors:", len(report["errors"]))
    print("Warnings:", len(report["warnings"]))

    for item in report["errors"][:10]:
        print("ERROR:", item)

    for item in report["warnings"][:10]:
        print("WARNING:", item)

    print("\nReport saved to:")
    print(REPORT_FILE)
