
import json
import hashlib
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.normalizer import normalize_arabic

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "dataset.jsonl"

VALID_DIALECTS = {
    "bahraini",
    "saudi",
    "kuwaiti",
    "emirati",
    "qatari",
    "omani",
    "msa",
}

DIALECT_COUNTRIES = {
    "bahraini": "Bahrain",
    "saudi": "Saudi Arabia",
    "kuwaiti": "Kuwait",
    "emirati": "United Arab Emirates",
    "qatari": "Qatar",
    "omani": "Oman",
    "msa": "Multiple",
}

# Metadata for this experimental batch only.
# Do not reuse these rights fields for external datasets.
SOURCE_METADATA = {
    "source": "user_authored",
    "license": "custom_author_owned",
    "consent_status": "author_created_for_project",
    "domain": "unspecified",
}

def get_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def prepare_dataset():
    records = []
    seen = set()

    stats = Counter()
    duplicate_count = 0

    if not RAW_DIR.exists():
        print("ERROR: Raw data directory not found:", RAW_DIR)
        return

    files = sorted(RAW_DIR.rglob("*.txt"))

    if not files:
        print("ERROR: No TXT files found in:", RAW_DIR)
        return

    for file_path in files:

        dialect_folder = file_path.parent.name.lower()

        if dialect_folder in VALID_DIALECTS:
            dialect = dialect_folder
        else:
            dialect = "unknown"

        country = DIALECT_COUNTRIES.get(
            dialect, "unknown"
        )

        content = file_path.read_text(
            encoding="utf-8"
        )

        for line in content.splitlines():

            original = line.strip()

            if len(original) < 5:
                continue

            normalized = normalize_arabic(original)

            if not normalized:
                continue

            fingerprint = get_hash(normalized)

            if fingerprint in seen:
                duplicate_count += 1
                continue

            seen.add(fingerprint)

            record = {
                "id": f"AGAI_{len(records) + 1:06d}",
                "text": original,
                "normalized_text": normalized,
                "language": "ar",
                "dialect": dialect,
                "country": country,
                "domain": SOURCE_METADATA["domain"],
                "source": SOURCE_METADATA["source"],
                "license": SOURCE_METADATA["license"],
                "consent_status": SOURCE_METADATA["consent_status"],
                "dialect_confidence": "folder_labeled",
                "source_file": str(
                    file_path.relative_to(ROOT)
                ),
                "normalization_version": "0.2"
            }

            records.append(record)
            stats[dialect] += 1

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n===== ARABIC GULF AI DATA PIPELINE v0.2 =====")
    print("Total records:", len(records))
    print("Duplicate records skipped:", duplicate_count)

    print("\nDialect distribution:")
    for dialect, count in sorted(stats.items()):
        print(f"{dialect}: {count}")

    print("\nOutput:", OUTPUT_FILE)
    print("Dataset preparation completed.")


if __name__ == "__main__":
    prepare_dataset()
