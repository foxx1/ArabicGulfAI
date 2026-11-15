import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.quality import (
    audit_dataset,
    print_report,
)

if __name__ == "__main__":
    report = audit_dataset()
    print_report(report)

    if report["status"] != "passed":
        sys.exit(1)
