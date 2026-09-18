"""Apply audited binary usability labels to the complete broadcast dataset."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from labeling.app import Dataset

OUT = ROOT / "pilot" / "output"
PREDICTIONS = OUT / "pool_predictions.json"
PILOT = OUT / "pilot_labels.json"
MANUAL_REVIEW = ROOT / "pilot" / "manual_binary_review.json"
DATASET = ROOT / "datasets" / "ice_v1"


def load_plan():
    payload = json.loads(PREDICTIONS.read_text())
    predictions = {record["filename"]: record for record in payload["records"]}
    pilot = {
        record["filename"]: record["model_label"]["position_usability"]
        for record in json.loads(PILOT.read_text())["records"]
    }
    manual = json.loads(MANUAL_REVIEW.read_text())["reviewed_predictions"]
    if set(manual) - set(predictions):
        raise ValueError("Manual review contains filenames absent from predictions")

    plan = {}
    for filename, prediction in predictions.items():
        final_label = prediction["prediction"]
        source = "descriptor_ensemble"
        if filename in pilot:
            final_label = pilot[filename]
            source = "approved_pilot"
        if filename in manual:
            final_label = manual[filename]
            source = "model_visual_review"
        plan[filename] = (final_label, source, prediction)
    return plan, payload


def main():
    parser = ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write labels; otherwise show a dry run")
    args = parser.parse_args()
    dataset = Dataset(DATASET)
    plan, payload = load_plan()
    broadcast = [entry for entry in dataset.entries.values() if entry["source_id"] == "broadcast_01"]
    filenames = [entry["relative_path"] for entry in broadcast]
    if len(broadcast) != 1204 or set(filenames) != set(plan):
        raise ValueError(f"Expected a one-to-one plan for 1204 broadcast frames; got {len(broadcast)} records and {len(plan)} labels")

    records = [(entry, dataset.get(entry["id"])) for entry in broadcast]
    preserved = [(entry, record) for entry, record in records if record["review"]["shot"] == "reviewed"]
    pending = [(entry, record) for entry, record in records if record["review"]["shot"] != "reviewed"]
    final_counts = Counter()
    source_counts = Counter()
    for entry, record in records:
        if record["review"]["shot"] == "reviewed":
            final_counts[record["shot"]["position_usability"]] += 1
            source_counts["existing_review"] += 1
        else:
            label, source, _ = plan[entry["relative_path"]]
            final_counts[label] += 1
            source_counts[source] += 1

    if not args.apply:
        print(json.dumps({"mode": "dry_run", "would_write": len(pending), "would_preserve": len(preserved),
                          "final_counts": final_counts, "sources": source_counts}, default=dict))
        return

    backup_dir = DATASET / "automation_backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"binary_before_{stamp}.jsonl"
    with backup.open("x") as handle:
        for _, record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    for entry, record in pending:
        label, source, prediction = plan[entry["relative_path"]]
        record["shot"]["position_usability"] = label
        record["review"]["shot"] = "reviewed"
        record.setdefault("automation", {})["binary_usability"] = {
            "schema_version": 1,
            "source": source,
            "prediction": prediction["prediction"],
            "final_label": label,
            "vote_fraction": prediction["vote_fraction"],
            "nearest_pilot_distance": prediction["nearest_pilot_distance"],
            "training_frames": payload["training_frames"],
            "cross_validation_accuracy": payload["cross_validation_accuracy"],
        }
        dataset.save(record)

    print(json.dumps({"mode": "applied", "written": len(pending), "preserved": len(preserved),
                      "final_counts": final_counts, "sources": source_counts,
                      "backup": str(backup)}, default=dict))


if __name__ == "__main__":
    main()
