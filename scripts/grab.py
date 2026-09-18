"""
Pull the frames you need to hand-click for a propagation drift test:
frame 0 (the seed) plus N checkpoints spread evenly across the WHOLE clip.

    python scripts/grab.py <video> [n_checkpoints=6]   ->  writes frames/<idx>.png

Then:
    python -m scripts.benchmark collect frames labels    # click anchors on each
    python -m vision.propagate run <video> labels/0.png.json --out overlay.mp4 \
           150:labels/150.png.json 300:labels/300.png.json ...
Only these frames need clicking; the full video feeds vision.propagate directly.
"""
import sys, os
import cv2

video = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6

cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
if total <= 0:
    raise SystemExit(f"could not read frame count from {video}")

# seed at 0, then n checkpoints evenly spaced through the end of the clip
targets = sorted({0} | {round(total * i / n) for i in range(1, n + 1)})
targets = [min(t, total - 1) for t in targets]

os.makedirs("frames", exist_ok=True)
want = set(targets)
i = 0
while want:
    ok, fr = cap.read()          # sequential read = exact frames (seeking is unreliable)
    if not ok:
        break
    if i in want:
        cv2.imwrite(f"frames/{i}.png", fr)
        print(f"wrote frames/{i}.png   (t = {i / fps:5.1f}s)")
        want.discard(i)
    i += 1
cap.release()

print(f"\nclip: {total} frames @ {fps:.2f} fps ({total/fps:.1f}s)")
print(f"seed        = frames/{targets[0]}.png")
print(f"checkpoints = " + " ".join(f"{t}:labels/{t}.png.json" for t in targets[1:]))
