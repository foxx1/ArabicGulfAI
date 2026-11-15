
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "dataset.jsonl"

def audit():

    if not DATA_FILE.exists():
        print("Dataset not found. Run prepare_data.py first.")
        return

    dialects = Counter()
    total = 0

    with DATA_FILE.open(
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(line)

            total += 1
            dialects[record["dialect"]] += 1

    print("=" * 40)
    print("ARABIC GULF DATA AUDIT")
    print("=" * 40)

    print("Total records:", total)

    print("\nDialect distribution:")

    for dialect, count in dialects.items():
        print(f"{dialect}: {count}")

    print("\nAudit completed.")

if __name__ == "__main__":
    audit()