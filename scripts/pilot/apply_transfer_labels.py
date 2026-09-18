"""Apply audited usability labels for newly added broadcast sources."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from labeling.app import Dataset

DATASET = ROOT / "datasets" / "ice_v1"
PREDICTIONS = ROOT / "pilot" / "output" / "transfer" / "retrained_predictions.json"
REVIEW = ROOT / "pilot" / "transfer_review.json"
SOURCE_IDS = ("min_vs_sjs", "tor_vs_tbl")


def main():
    parser = ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    predictions = json.loads(PREDICTIONS.read_text())
    review = json.loads(REVIEW.read_text())
    post = review["post_retrain_corrections"]
    dataset = Dataset(DATASET)
    targets = [entry for entry in dataset.entries.values() if entry["source_id"] in SOURCE_IDS]
    records = [(entry, dataset.get(entry["id"])) for entry in targets]
    plan = {}
    for source_id in SOURCE_IDS:
        for proposal in predictions["sources"][source_id]["records"]:
            label = post.get(source_id, {}).get(proposal["filename"], proposal["prediction"])
            if proposal["transfer_pilot"]:
                source = "transfer_pilot_visual_review"
            elif proposal.get("review_group"):
                source = "model_visual_review"
            else:
                source = "three_broadcast_descriptor_ensemble"
            plan[(source_id, proposal["filename"])] = label, source, proposal
    expected = {(entry["source_id"], entry["relative_path"]) for entry in targets}
    if len(targets) != 510 or expected != set(plan):
        raise ValueError(f"Expected a one-to-one plan for 510 frames; got {len(targets)} records and {len(plan)} labels")

    pending = [(entry, record) for entry, record in records if record["review"]["shot"] != "reviewed"]
    preserved = len(records) - len(pending)
    counts, sources = Counter(), Counter()
    for entry, record in records:
        if record["review"]["shot"] == "reviewed":
            counts[record["shot"]["position_usability"]] += 1
            sources["existing_review"] += 1
        else:
            label, source, _ = plan[(entry["source_id"], entry["relative_path"])]
            counts[label] += 1
            sources[source] += 1
    summary = {"would_write": len(pending), "would_preserve": preserved,
               "final_counts": dict(counts), "sources": dict(sources)}
    if not args.apply:
        print(json.dumps({"mode": "dry_run", **summary}))
        return

    backup_dir = DATASET / "automation_backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"transfer_before_{stamp}.jsonl"
    with backup.open("x") as handle:
        for _, record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    for entry, record in pending:
        label, source, proposal = plan[(entry["source_id"], entry["relative_path"])]
        record["shot"]["position_usability"] = label
        record["review"]["shot"] = "reviewed"
        record.setdefault("automation", {})["binary_usability"] = {
            "schema_version": 1,
            "source": source,
            "prediction": proposal["prediction"],
            "final_label": label,
            "vote_fraction": proposal["vote_fraction"],
            "training_frames": predictions["training_frames"],
            "training_broadcasts": 3,
            "review_group": proposal.get("review_group"),
        }
        dataset.save(record)
    print(json.dumps({"mode": "applied", **summary, "backup": str(backup)}))


if __name__ == "__main__":
    main()
