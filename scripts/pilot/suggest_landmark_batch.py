"""Draft landmark proposals for the unreviewed geometry-pilot frames.

This is intentionally a review accelerator rather than an automatic labeler. It
finds the closest reviewed view from the same broadcast, aligns the ice surfaces
with local image features, transfers the reviewed landmarks, and snaps dot-like
landmarks to nearby red paint blobs. Proposals remain unreviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "ice_v1"

def read(path: Path):
    return json.loads(path.read_text())


def descriptor(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, _, red = cv2.split(image)
    blue, red = blue.astype(float), red.astype(float)
    ice = ((hsv[:, :, 2] > 130) & (hsv[:, :, 1] < 100)).astype("uint8") * 255
    relred = np.clip((red - blue - 5) * 5, 0, 255).astype("uint8")
    relblue = np.clip((blue - red - 5) * 5, 0, 255).astype("uint8")
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 70, 150)
    return np.concatenate([
        cv2.resize(channel, (48, 27), interpolation=cv2.INTER_AREA).ravel() / 255
        for channel in (ice, relred, relblue, edges)
    ]).astype("float32")


def red_dot_candidates(image):
    blue, _, red = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((red.astype(int) - blue.astype(int) > 8)
            & (hsv[:, :, 2] > 100)).astype("uint8")
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    found = []
    for stat, center in zip(stats[1:count], centers[1:count]):
        x, y, width, height, area = stat
        ratio = width / max(height, 1)
        fill = area / max(width * height, 1)
        if y >= 230 and 120 < area < 3000 and 1.4 < ratio < 7 and fill > .4:
            found.append(np.asarray(center, dtype=float))
    return found


def paint_distances(image):
    blue, _, red = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ice_or_paint = (hsv[:, :, 2] > 80) & (hsv[:, :, 1] < 190)
    red_mask = ((red.astype(int) - blue.astype(int) > 5) & ice_or_paint).astype("uint8")
    blue_mask = ((blue.astype(int) - red.astype(int) > 8) & ice_or_paint).astype("uint8")
    return (cv2.distanceTransform(1 - red_mask, cv2.DIST_L2, 3),
            cv2.distanceTransform(1 - blue_mask, cv2.DIST_L2, 3))


def snap(name, point, candidates):
    if not (name.startswith("dot") or name == "center") or not candidates:
        return point
    distances = [np.linalg.norm(point - candidate) for candidate in candidates]
    index = int(np.argmin(distances))
    return candidates[index] if distances[index] <= 110 else point


def reviewed_features(record):
    unique = {}
    for feature in record["features"]:
        name = feature.get("landmark")
        if name and len(feature.get("points", [])) == 1 and name not in unique:
            unique[name] = feature
    return list(unique.values())


def sift_features(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 100) & (hsv[:, :, 1] < 150)).astype("uint8") * 255
    mask[:180] = 0
    return sift.detectAndCompute(gray, mask)


def align_views(reference, target, sift_cache):
    reference_points, reference_descriptors = sift_cache[reference["id"]]
    target_points, target_descriptors = sift_cache[target["id"]]
    if reference_descriptors is None or target_descriptors is None:
        return None, {"matches": 0, "inliers": 0}
    matches = cv2.BFMatcher().knnMatch(reference_descriptors, target_descriptors, k=2)
    good = [first for first, second in matches if first.distance < .72 * second.distance]
    if len(good) < 12:
        return None, {"matches": len(good), "inliers": 0}
    source = np.float32([reference_points[match.queryIdx].pt for match in good])
    destination = np.float32([target_points[match.trainIdx].pt for match in good])
    transform, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5)
    inliers = int(mask.sum()) if mask is not None else 0
    if transform is None or inliers < 10 or inliers / len(good) < .15:
        return None, {"matches": len(good), "inliers": inliers}
    return transform, {"matches": len(good), "inliers": inliers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--source", help="Only draft frames from this broadcast source")
    parser.add_argument("--reference-scope", choices=["same-source", "all"],
                        default="same-source",
                        help="Choose references from the target broadcast or all broadcasts")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest = read(dataset / "manifest.json")
    by_id = {entry["id"]: entry for entry in manifest["images"]}
    pilot_ids = manifest["selections"]["geometry_pilot_v1"]
    target_ids = [image_id for image_id in pilot_ids
                  if not args.source or by_id[image_id]["source_id"] == args.source]

    records = {image_id: read(dataset / "records" / f"{image_id}.json")
               for image_id in pilot_ids}
    images = {image_id: cv2.imread(by_id[image_id]["path"]) for image_id in pilot_ids}
    features = {image_id: descriptor(images[image_id]) for image_id in pilot_ids}

    source_entries = {}
    normalized_features = {}
    for image_id in pilot_ids:
        group = (by_id[image_id]["source_id"]
                 if args.reference_scope == "same-source" else "all")
        source_entries.setdefault(group, []).append(by_id[image_id])
    for source, entries in source_entries.items():
        matrix = np.asarray([features[entry["id"]] for entry in entries], np.float32)
        matrix = (matrix - matrix.mean(0)) / np.where(matrix.std(0) > .03, matrix.std(0), 1)
        normalized_features.update({entry["id"]: row for entry, row in zip(entries, matrix)})

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.015)
    sift_cache = {image_id: sift_features(images[image_id], sift) for image_id in pilot_ids}

    reviewed_by_source = {}
    for image_id in pilot_ids:
        entry, record = by_id[image_id], records[image_id]
        if (record["review"].get("landmarks") == "reviewed"
                and record.get("shot", {}).get("position_usability") == "usable"):
            observed = reviewed_features(record)
            if observed:
                item = (entry, record, normalized_features[image_id])
                reviewed_by_source.setdefault(entry["source_id"], []).append(item)

    reviewed_all = [item for items in reviewed_by_source.values() for item in items]
    drafts = {}
    abstained = 0
    for image_id in target_ids:
        record, entry = records[image_id], by_id[image_id]
        if record["review"].get("landmarks", "unreviewed") != "unreviewed":
            continue
        source = entry["source_id"]
        reference = None
        medoid_name = None
        alignment = None
        alignment_diagnostics = {"matches": 0, "inliers": 0}
        if record.get("shot", {}).get("position_usability") != "usable":
            drafts[image_id] = ([], source, medoid_name, None, alignment_diagnostics)
            abstained += 1
            continue
        references = (reviewed_by_source.get(source, [])
                      if args.reference_scope == "same-source" else reviewed_all)
        if references:
            target = normalized_features[image_id]
            reference = min(references,
                            key=lambda item: np.mean((item[2] - target) ** 2))
            alignment, alignment_diagnostics = align_views(reference[0], entry, sift_cache)
        if reference is None:
            drafts[image_id] = ([], source, medoid_name, None, alignment_diagnostics)
            abstained += 1
            continue

        reference_entry, reference_record, _ = reference
        if alignment is None:
            drafts[image_id] = ([], source, medoid_name,
                                Path(reference_entry["path"]).name,
                                alignment_diagnostics)
            abstained += 1
            continue
        candidates = red_dot_candidates(images[image_id])
        red_distance, blue_distance = paint_distances(images[image_id])
        proposals = []
        for feature in reviewed_features(reference_record):
            name = feature["landmark"]
            point = np.asarray(feature["points"][0], dtype="float32")
            point = cv2.perspectiveTransform(point.reshape(1, 1, 2), alignment).reshape(2)
            point = snap(name, point.astype(float), candidates)
            x, y = point
            if not (0 <= x < entry["width"] and 180 <= y < entry["height"] - 10):
                continue
            px, py = int(round(x)), int(round(y))
            distance = blue_distance[py, px] if name.startswith("blue") else red_distance[py, px]
            if distance > 50:
                continue
            token = hashlib.sha256(f"batch-v1:{image_id}:{name}".encode()).hexdigest()[:16]
            proposals.append({
                "id": f"batch-v1-{token}", "type": "landmark", "landmark": name,
                "points": [[float(x), float(y)]], "visibility": "visible",
                "provenance": "model_inferred", "held_out": False,
            })
        drafts[image_id] = (proposals, source, medoid_name, Path(reference_entry["path"]).name,
                            alignment_diagnostics)

    total = sum(len(value[0]) for value in drafts.values())
    trapezoids = sum(p["landmark"].startswith("trap") for value in drafts.values() for p in value[0])
    summary = {"frames": len(drafts), "proposals": total,
               "trapezoid_proposals": trapezoids, "abstained_frames": abstained}
    if not args.apply:
        print(json.dumps(summary, indent=2))
        return

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = dataset / "automation_backups" / f"landmark_batch_before_{stamp}.jsonl"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("".join(json.dumps(records[image_id]) + "\n" for image_id in drafts))
    for image_id, value in drafts.items():
        proposals, source, medoid_name, template_name = value[:4]
        alignment_diagnostics = value[4] if len(value) > 4 else {"matches": 0, "inliers": 0}
        record = records[image_id]
        # A rerun replaces only this script's drafts and preserves proposals from
        # other detectors/imports.
        record["proposals"] = [proposal for proposal in record["proposals"]
                               if not proposal.get("id", "").startswith("batch-v1-")]
        existing = {feature.get("landmark") for feature in record["features"] + record["proposals"]}
        added = [proposal for proposal in proposals if proposal["landmark"] not in existing]
        record["proposals"].extend(added)
        record.setdefault("automation", {})["landmark_draft"] = {
            "source": "aligned_reviewed_view_v2",
            "reference_scope": args.reference_scope,
            "target_source_filter": args.source,
            "status": "requires_human_verification",
            "proposal_count": len(added),
            "template_frame": template_name,
            "view_medoid": medoid_name,
            "alignment_matches": alignment_diagnostics["matches"],
            "alignment_inliers": alignment_diagnostics["inliers"],
            "trapezoid_candidates_included": any(p["landmark"].startswith("trap") for p in added),
        }
        record["revision"] += 1
        target = dataset / "records" / f"{image_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(target)
    summary["backup"] = str(backup)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
