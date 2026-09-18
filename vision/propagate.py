"""Propagate a labeled rink homography through video using optical flow.

Seed a homography once (from vision.rink-style clicked anchors on frame 0 of a shot),
then carry it forward frame-to-frame with Lucas-Kanade optical flow on static ice
features. Each frame re-solves image->rink from tracked points + their stored rink
coords (NOT blind matrix chaining -> limits drift), applies a validity gate, and
replenishes points as the camera pans.

    python -m vision.propagate run <video> <seed.json> [--out overlay.mp4] \
                           [--checkpoint FRAME:label.json ...]
    python -m vision.propagate selftest      # synthetic validation, no footage needed

seed.json / checkpoint label.json use vision.rink's format: {landmark_name: [px, py]}.
"""
import sys, os, json
import numpy as np
import cv2
from vision.rink import LANDMARKS

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class HomographyTracker:
    def __init__(self, min_inliers=6, max_resid_ft=3.0, fb_thresh=1.0,
                 target_pts=180, ransac_ft=3.0, min_anchors=4):
        self.min_inliers, self.max_resid_ft = min_inliers, max_resid_ft
        self.fb_thresh, self.target_pts, self.ransac_ft = fb_thresh, target_pts, ransac_ft
        # min_anchors: require this many ORIGINAL clicked anchors still tracked to
        # trust the fix. Without it, drift on replenished points reads as "valid".
        self.min_anchors = min_anchors
        self.prev = None                 # previous grayscale frame
        self.pts = None                  # Nx2 float32 image points (current frame)
        self.rink = None                 # Nx2 float32 rink coords aligned to pts
        self.is_anchor = None            # bool: exact clicked anchor (vs replenished)
        self.H = None                    # last VALID image->rink homography (held on no-fix)
        self.status = "no fix"

    # ---- seeding -------------------------------------------------------------
    def seed(self, frame, anchor_img, anchor_rink):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ai = np.asarray(anchor_img, np.float32)
        ar = np.asarray(anchor_rink, np.float32)
        H, _ = cv2.findHomography(ai, ar, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("seed failed: anchors are degenerate (need >=4 non-collinear)")
        self.H, self.status = H, "valid"
        self.prev = gray
        self.pts, self.rink = ai.copy(), ar.copy()
        self.is_anchor = np.ones(len(ai), bool)
        self._replenish(gray)
        return H

    def _replenish(self, gray):
        if self.H is None:
            return
        need = self.target_pts - len(self.pts)
        if need <= 20:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for x, y in self.pts.astype(int):
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                cv2.circle(mask, (int(x), int(y)), 8, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=int(need), qualityLevel=0.01,
                                      minDistance=10, mask=mask)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        rink_new = cv2.perspectiveTransform(new.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        self.pts = np.vstack([self.pts, new])
        self.rink = np.vstack([self.rink, rink_new])
        self.is_anchor = np.concatenate([self.is_anchor, np.zeros(len(new), bool)])

    # ---- per-frame update ----------------------------------------------------
    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diag = {"tracked": 0, "inliers": 0, "med_resid_ft": None,
                "anchors_live": 0, "status": "no fix"}
        if self.prev is None or self.pts is None or len(self.pts) < 4:
            self.status = "no fix"; diag["status"] = self.status
            self.prev = gray
            return None, self.status, diag

        p0 = self.pts.reshape(-1, 1, 2)
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **LK)
        p0b, s2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **LK)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        good = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < self.fb_thresh)

        self.pts = p1.reshape(-1, 2)[good]
        self.rink = self.rink[good]
        self.is_anchor = self.is_anchor[good]
        diag["tracked"] = int(good.sum())
        diag["anchors_live"] = int(self.is_anchor.sum())

        if len(self.pts) < 4:
            self.prev = gray; self.status = "no fix"; diag["status"] = self.status
            return None, self.status, diag

        H, inl = cv2.findHomography(self.pts, self.rink, cv2.RANSAC, self.ransac_ft)
        self.prev = gray
        if H is None:
            self.status = "no fix"; diag["status"] = self.status
            return None, self.status, diag

        inl = inl.ravel().astype(bool)
        proj = cv2.perspectiveTransform(self.pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        resid = np.linalg.norm(proj - self.rink, axis=1)
        n_in = int(inl.sum())
        med = float(np.median(resid[inl])) if n_in else 1e9
        diag["inliers"], diag["med_resid_ft"] = n_in, round(med, 2)

        if (n_in >= self.min_inliers and med <= self.max_resid_ft
                and diag["anchors_live"] >= self.min_anchors):
            self.H, self.status = H, "valid"      # accept & update held homography
        else:
            self.status = "no fix"                # keep previous self.H as the hold
        diag["status"] = self.status
        self._replenish(gray)
        return (self.H if self.status == "valid" else None), self.status, diag


# ---- rink overlay (reproject the model onto the frame to SEE drift) ----------
_DOTS = [(0, 0), (20, 22), (20, -22), (-20, 22), (-20, -22),
         (69, 22), (69, -22), (-69, 22), (-69, -22)]
_LINES = [((-89, -36.75), (-89, 36.75)), ((89, -36.75), (89, 36.75)),   # goal lines
          ((-25, -42.5), (-25, 42.5)), ((25, -42.5), (25, 42.5)),       # blue lines
          ((0, -42.5), (0, 42.5))]                                       # center line

def draw_rink_overlay(frame, H, status="valid"):
    out = frame.copy()
    if H is not None:
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return out
        def to_img(pts):
            p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(p, Hinv).reshape(-1, 2)
        for a, b in _LINES:
            pa, pb = to_img([a, b]).astype(int)
            cv2.line(out, tuple(pa), tuple(pb), (0, 255, 255), 2)
        for c in to_img(_DOTS).astype(int):
            cv2.circle(out, tuple(c), 4, (0, 0, 255), -1)
    color = (0, 200, 0) if status == "valid" else (0, 0, 255)
    cv2.putText(out, status.upper(), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def checkpoint_error_ft(H, label_json):
    """True drift: map clicked landmark pixels through H, compare to their rink coords."""
    lab = {k: v for k, v in json.load(open(label_json)).items() if k in LANDMARKS}
    if H is None or not lab:
        return None
    img = np.array([lab[k] for k in lab], np.float32)
    rink = np.array([LANDMARKS[k] for k in lab], np.float32)
    proj = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.median(np.linalg.norm(proj - rink, axis=1)))


# ---- driver -----------------------------------------------------------------
def run(video, seed_json, out=None, checkpoints=None):
    checkpoints = checkpoints or {}
    seed = {k: v for k, v in json.load(open(seed_json)).items() if k in LANDMARKS}
    if len(seed) < 4:
        raise ValueError(f"seed has {len(seed)} anchors; need >=4")
    ai = [seed[k] for k in seed]
    ar = [LANDMARKS[k] for k in seed]

    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"cannot read {video}")
    trk = HomographyTracker()
    trk.seed(frame, ai, ar)

    writer = None
    if out:
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(draw_rink_overlay(frame, trk.H, "valid"))

    idx, nofix = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        H, status, diag = trk.update(frame)
        if status != "valid":
            nofix += 1
        if writer:
            writer.write(draw_rink_overlay(frame, trk.H, status))
        if idx in checkpoints:
            err = checkpoint_error_ft(trk.H, checkpoints[idx])
            print(f"  checkpoint frame {idx}: drift {err:.2f} ft  "
                  f"(status={status}, inliers={diag['inliers']}, anchors_live={diag['anchors_live']})")
    cap.release()
    if writer:
        writer.release()
    total = idx + 1
    print(f"\nframes: {total}  no-fix: {nofix} ({100*nofix/max(total,1):.0f}%)"
          + (f"  overlay -> {out}" if out else ""))


# ---- synthetic self-test (validates the core without hockey footage) --------
def _make_texture():
    """A rink-like plane in 'rink-pixel' coords: S px/ft, offset. Lines+dots+speckle."""
    S, ox, oy = 5, 520, 240
    tex = np.full((480, 1040, 3), 40, np.uint8)
    def rp(x, y): return (int(x * S + ox), int(y * S + oy))
    rng = np.random.default_rng(0)
    for _ in range(1500):                      # speckle so LK has features
        x, y = rng.uniform(-100, 100), rng.uniform(-42.5, 42.5)
        cv2.circle(tex, rp(x, y), 1, (110, 110, 110), -1)
    for a, b in _LINES:
        cv2.line(tex, rp(*a), rp(*b), (200, 120, 120), 2)
    for d in _DOTS:
        cv2.circle(tex, rp(*d), 4, (80, 80, 220), -1)
    def to_rinkpx(ft): return np.array([ft[0] * S + ox, ft[1] * S + oy], np.float32)
    return tex, to_rinkpx

def selftest():
    tex, to_rinkpx = _make_texture()
    W, H = 1280, 720
    src = np.float32([to_rinkpx((-100, -42.5)), to_rinkpx((100, -42.5)),
                      to_rinkpx((100, 42.5)), to_rinkpx((-100, 42.5))])

    def cam(t):                                # smooth pan + slight zoom over t in [0,1]
        pan = 260 * t
        z = 1.0 + 0.15 * t
        cx, cy = W / 2 + pan, H / 2
        w2, h2 = 560 * z, 300 * z
        dst = np.float32([[cx - w2, cy - h2], [cx + w2, cy - h2],
                          [cx + w2, cy + h2], [cx - w2, cy + h2]])
        return cv2.getPerspectiveTransform(src, dst)   # rinkpx -> image

    N = 120
    lm_names = list(LANDMARKS)
    lm_rinkpx = np.array([to_rinkpx(LANDMARKS[n]) for n in lm_names], np.float32)
    lm_rink_ft = np.array([LANDMARKS[n] for n in lm_names], np.float32)

    def frame_at(t):
        return cv2.warpPerspective(tex, cam(t), (W, H))
    def lm_image_at(t):
        return cv2.perspectiveTransform(lm_rinkpx.reshape(-1, 1, 2), cam(t)).reshape(-1, 2)

    # seed on frame 0 with a realistic subset of visible anchors
    img0 = lm_image_at(0.0)
    vis = [i for i, (x, y) in enumerate(img0) if 0 <= x < W and 0 <= y < H][:8]
    trk = HomographyTracker()
    trk.seed(frame_at(0.0), img0[vis], lm_rink_ft[vis])

    drift = []
    for k in range(1, N + 1):
        t = k / N
        _, status, _ = trk.update(frame_at(t))
        if trk.H is not None:
            proj = cv2.perspectiveTransform(lm_image_at(t).reshape(-1, 1, 2),
                                            trk.H).reshape(-1, 2)
            # only score landmarks currently in view
            imv = lm_image_at(t)
            m = (imv[:, 0] >= 0) & (imv[:, 0] < W) & (imv[:, 1] >= 0) & (imv[:, 1] < H)
            drift.append(np.median(np.linalg.norm(proj[m] - lm_rink_ft[m], axis=1)))
    drift = np.array(drift)
    print(f"synthetic pan+zoom, {N} frames, seeded with {len(vis)} anchors")
    print(f"  drift vs ground truth  ->  median {np.median(drift):.2f} ft   "
          f"p90 {np.percentile(drift,90):.2f} ft   max {drift.max():.2f} ft")
    ok = np.median(drift) < 1.0 and drift.max() < 3.0
    print("  RESULT:", "PASS (propagation core tracks the plane)" if ok
          else "FAIL (drift too high)")
    return ok


def _parse_checkpoints(args):
    cps = {}
    for a in args:
        if a.startswith("--checkpoint"):
            continue
        if ":" in a and a.split(":")[0].isdigit():
            f, path = a.split(":", 1)
            cps[int(f)] = path
    return cps

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(0 if selftest() else 1)
    elif len(sys.argv) >= 4 and sys.argv[1] == "run":
        out = None
        if "--out" in sys.argv:
            out = sys.argv[sys.argv.index("--out") + 1]
        cps = _parse_checkpoints(sys.argv[4:])
        run(sys.argv[2], sys.argv[3], out=out, checkpoints=cps)
    else:
        print(__doc__)
