"""Detect rink paint and landmarks in individual hockey broadcast frames.

Key design (each layer kills a failure mode observed on real footage):
  1. ICE HULL: detect only inside the bright low-sat ice region (kills crowd/scorebug/board ads)
  2. RELATIVE COLOR: classify vs the ice's own white balance (B-R margin), not absolute
     saturation — broadcast blue lines measured at S~5 absolute but +30..40 relative.
  3. STRUCTURE: rink lines are LONG (big collinear support) and THIN (perpendicular
     run test) — rejects center-ice logos, jerseys, on-ice ads of the same color.
  4. DOTS must be surrounded by open ice (ring test).

    python -m vision.detect show  <image.png>
    python -m vision.detect score <frames_dir> <labels_dir>
    python -m vision.detect tune  <image.png>
"""
import sys, os, glob, json
import numpy as np
import cv2
from vision.rink import LANDMARKS

# measured on real broadcast frame (Stanley Cup Final @ Carolina):
REL_BLUE_TH = 15      # blue line measured +30..42 vs ice
REL_RED_TH  = 14      # center line measured +16..51 vs ice
HULL_V, HULL_S = 150, 50

MODEL_LINES = [  # (rink endpoints, color)
    (((0, -42.5), (0, 42.5)), "red"),
    (((-25, -42.5), (-25, 42.5)), "blue"), (((25, -42.5), (25, 42.5)), "blue"),
    (((-89, -36.75), (-89, 36.75)), "red"), (((89, -36.75), (89, 36.75)), "red"),
]
MODEL_DOTS = [(0,0),(20,22),(20,-22),(-20,22),(-20,-22),(69,22),(69,-22),(-69,22),(-69,-22)]

def _ice_hull(hsv):
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    m = ((V > HULL_V) & (S < HULL_S)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros(hsv.shape[:2], bool)
    big = 1 + np.argmax(stats[1:, 4])
    pts = cv2.convexHull(np.argwhere(lab == big)[:, ::-1].astype(np.int32))
    hull = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(hull, [pts], 255)
    return cv2.erode(hull, np.ones((15, 15), np.uint8)) > 0

def _merge_lines(segs, dist_tol=10, ang_tol=6):
    """Greedy-merge Hough segments into long lines; return fitted lines with support."""
    segs = sorted(segs, key=lambda s: -np.hypot(s[2]-s[0], s[3]-s[1]))
    used = [False]*len(segs); out = []
    def ang(s): return np.arctan2(s[3]-s[1], s[2]-s[0]) % np.pi
    for i, s in enumerate(segs):
        if used[i]: continue
        member_pts = [(s[0],s[1]),(s[2],s[3])]; used[i] = True; a0 = ang(s)
        p = np.array([(s[0]+s[2])/2,(s[1]+s[3])/2],float)
        d = np.array([np.cos(a0),np.sin(a0)])
        for j, t in enumerate(segs):
            if used[j]: continue
            da = abs(ang(t)-a0); da = min(da, np.pi-da)
            if da > np.deg2rad(ang_tol): continue
            n = np.array([-d[1], d[0]])
            if max(abs(np.dot(np.array([t[0],t[1]],float)-p, n)),
                   abs(np.dot(np.array([t[2],t[3]],float)-p, n))) > dist_tol: continue
            member_pts += [(t[0],t[1]),(t[2],t[3])]; used[j] = True
            P = np.array(member_pts, np.float32)
            vx,vy,x0,y0 = cv2.fitLine(P, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            d = np.array([vx,vy]); p = np.array([x0,y0]); a0 = np.arctan2(vy,vx)%np.pi
        P = np.array(member_pts, float)
        proj = (P - p) @ d
        support = sum(np.hypot(segs[k][2]-segs[k][0], segs[k][3]-segs[k][1])
                      for k in range(len(segs)) if used[k] and
                      any(np.allclose([segs[k][0],segs[k][1]], mp) for mp in member_pts[:1])) \
                  if False else 0
        # support = total member segment length (recompute simply)
        support = 0.0
        for (x1,y1),(x2,y2) in zip(member_pts[0::2], member_pts[1::2]):
            support += np.hypot(x2-x1, y2-y1)
        e0, e1 = p + proj.min()*d, p + proj.max()*d
        out.append(dict(p0=tuple(e0), p1=tuple(e1), dir=d, point=p, support=support,
                        extent=float(proj.max()-proj.min())))
    return out

def _thickness_ok(line, mask, max_thick):
    p, d = np.array(line["point"]), np.array(line["dir"])
    n = np.array([-d[1], d[0]])
    L = line["extent"]; runs = []
    for t in np.linspace(-L/2*0.9, L/2*0.9, 15):
        c = p + t*d
        x, y = int(round(c[0])), int(round(c[1]))
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
            continue
        run = 1
        for sgn in (1, -1):
            for k in range(1, 60):
                q = c + sgn*k*n
                xi, yi = int(round(q[0])), int(round(q[1]))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                    run += 1
                else: break
        runs.append(run)
    return len(runs) >= 4 and np.median(runs) <= max_thick

def detect(frame, exclude=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hull = _ice_hull(hsv)
    B, R = frame[:, :, 0].astype(int), frame[:, :, 2].astype(int)
    ok = hull if exclude is None else (hull & ~exclude)
    base = float(np.median((B - R)[ok])) if ok.sum() > 100 else 0.0
    relblue = np.where(ok, (B - R) - base, 0)
    relred  = np.where(ok, (R - B) + base, 0)
    mb = cv2.morphologyEx(((relblue > REL_BLUE_TH)).astype(np.uint8)*255,
                          cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mr = cv2.morphologyEx(((relred  > REL_RED_TH )).astype(np.uint8)*255,
                          cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    scale = np.sqrt(hull.sum()) if hull.any() else 1
    min_support, max_thick = 0.15*scale, 0.030*scale
    min_extent = 0.30*scale
    distT = cv2.distanceTransform(hull.astype(np.uint8), cv2.DIST_L2, 3)
    h, w = hull.shape

    def continuity(ln, mask):
        p, d = np.array(ln["point"]), np.array(ln["dir"]); L = ln["extent"]
        on = 0; n = 40
        for t in np.linspace(-L/2, L/2, n):
            c = p + t*d; x, y = int(round(c[0])), int(round(c[1]))
            if 0 <= y < h and 0 <= x < w and mask[max(0,y-2):y+3, max(0,x-2):x+3].any():
                on += 1
        return on / n

    def ends_at_boards(ln):
        near = 0
        for e in (ln["p0"], ln["p1"]):
            x, y = int(round(e[0])), int(round(e[1]))
            edge = x < 12 or y < 12 or x > w-13 or y > h-13
            inb = 0 <= y < h and 0 <= x < w
            if edge or (inb and distT[y, x] < 0.08*scale) or not inb:
                near += 1
        return near >= 1   # at least one end reaches boards/frame edge

    lines = []
    for mask, color in ((mb, "blue"), (mr, "red")):
        segs = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=40,
                               minLineLength=40, maxLineGap=25)
        segs = [] if segs is None else [tuple(s[0]) for s in segs]
        for ln in _merge_lines(segs):
            if (ln["support"] >= min_support and ln["extent"] >= min_extent
                    and _thickness_ok(ln, mask > 0, max_thick)
                    and continuity(ln, mask > 0) >= 0.5
                    and ends_at_boards(ln)):
                ln["color"] = color; lines.append(ln)
    # dedup: collapse near-coincident fits of the same physical line (keep longest)
    lines.sort(key=lambda l: -l["extent"])
    kept = []
    for l in lines:
        dup = False
        for k in kept:
            if l["color"] != k["color"]: continue
            da = abs(np.arctan2(l["dir"][1], l["dir"][0]) % np.pi
                     - np.arctan2(k["dir"][1], k["dir"][0]) % np.pi)
            da = min(da, np.pi - da)
            if da < np.deg2rad(4) and _seg_dist(l["point"], k) < 18:
                dup = True; break
        if not dup: kept.append(l)
    lines = kept

    ice_like = (hsv[:,:,2] > HULL_V) & (hsv[:,:,1] < HULL_S+10)
    dots = []
    cnts, _ = cv2.findContours(mr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if not (15 <= a <= 1200): continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0 or a/(np.pi*r*r) < 0.5: continue
        if not (0 <= int(y) < hull.shape[0] and 0 <= int(x) < hull.shape[1] and hull[int(y),int(x)]): continue
        ann = np.zeros(hull.shape, np.uint8)
        cv2.circle(ann,(int(x),int(y)),int(3.2*r),255,-1); cv2.circle(ann,(int(x),int(y)),int(1.9*r),0,-1)
        am = ann > 0
        if am.sum() < 20 or ice_like[am].mean() < 0.5: continue
        near_line = any(abs(np.dot(np.array([x,y])-np.array(l["point"]),
                        np.array([-l["dir"][1], l["dir"][0]]))) < 10 and
                        abs(np.dot(np.array([x,y])-np.array(l["point"]), np.array(l["dir"]))) < l["extent"]/2+10
                        for l in lines)
        if not near_line:
            dots.append((float(x), float(y)))
    return dict(hull=hull, relblue=relblue, relred=relred, mb=mb, mr=mr,
                lines=lines, dots=dots)

def overlay(frame, det):
    out = frame.copy()
    hb = cv2.dilate(det["hull"].astype(np.uint8)*255, np.ones((3,3),np.uint8)) - det["hull"].astype(np.uint8)*255
    out[hb > 0] = (0, 255, 0)
    for l in det["lines"]:
        c = (255, 160, 0) if l["color"] == "blue" else (255, 0, 255)
        cv2.line(out, tuple(np.int32(l["p0"])), tuple(np.int32(l["p1"])), c, 3)
    for x, y in det["dots"]:
        cv2.circle(out, (int(x), int(y)), 9, (0, 255, 255), 2)
    nb = sum(1 for l in det["lines"] if l["color"]=="blue"); nr = len(det["lines"])-nb
    cv2.putText(out, f"lines: {nb} blue {nr} red   dots: {len(det['dots'])}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out

def show(path, exclude=None):
    fr = cv2.imread(path); assert fr is not None, path
    det = detect(fr, exclude)
    out = os.path.splitext(path)[0] + "_v2.png"
    cv2.imwrite(out, overlay(fr, det))
    nb = sum(1 for l in det["lines"] if l["color"]=="blue")
    print(f"{os.path.basename(path)}: lines {nb} blue {len(det['lines'])-nb} red, "
          f"dots {len(det['dots'])} -> {out}")
    return det

def tune(path):
    fr = cv2.imread(path); det = detect(fr); b = os.path.splitext(path)[0]
    cv2.imwrite(b+"_hull.png", det["hull"].astype(np.uint8)*255)
    cv2.imwrite(b+"_relblue.png", np.clip(det["relblue"]*4,0,255).astype(np.uint8))
    cv2.imwrite(b+"_relred.png",  np.clip(det["relred"]*4,0,255).astype(np.uint8))
    print(f"wrote {b}_hull/_relblue/_relred.png")

def _seg_dist(pt, l):
    p, d = np.array(l["point"]), np.array(l["dir"])
    return abs(np.dot(np.array(pt, float) - p, np.array([-d[1], d[0]])))

def score(frames_dir, labels_dir):
    tot = dict(line_vis=0, line_hit=0, det_lines=0, det_line_ok=0,
               dot_vis=0, dot_hit=0, det_dots=0, det_dot_ok=0)
    print(f"{'frame':26s} {'lines det/hit/vis':>18s} {'dots det/hit/vis':>18s}")
    for lp in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        base = os.path.basename(lp)[:-5]
        img = os.path.join(frames_dir, base)
        if not os.path.exists(img): continue
        lab = {k: v for k, v in json.load(open(lp)).items() if k in LANDMARKS}
        if len(lab) < 4: print(f"{base:26s}  skip (<4 clicks for GT)"); continue
        fr = cv2.imread(img)
        if fr is None: continue
        H_, _ = cv2.findHomography(np.float32([lab[k] for k in lab]),
                                   np.float32([LANDMARKS[k] for k in lab]), 0)
        Hi = np.linalg.inv(H_)
        h, w = fr.shape[:2]
        det = detect(fr)
        def toimg(pts):
            return cv2.perspectiveTransform(np.float32(pts).reshape(-1,1,2), Hi).reshape(-1,2)
        vis_lines = []
        for (a, b2), color in MODEL_LINES:
            ts = np.linspace(0, 1, 60)
            samp = toimg([(a[0]+(b2[0]-a[0])*t, a[1]+(b2[1]-a[1])*t) for t in ts])
            inview = samp[(samp[:,0]>=0)&(samp[:,0]<w)&(samp[:,1]>=0)&(samp[:,1]<h)]
            if len(inview) >= 8 and np.hypot(*(inview[-1]-inview[0])) > 120:
                vis_lines.append((inview, color))
        lv, lh = len(vis_lines), 0
        matched = set()
        for samp, color in vis_lines:
            best = None
            for idx, l in enumerate(det["lines"]):
                if l["color"] != color: continue
                dmed = np.median([_seg_dist(pt, l) for pt in samp])
                if dmed <= 12 and (best is None or dmed < best[0]): best = (dmed, idx)
            if best: lh += 1; matched.add(best[1])
        dl, dlok = len(det["lines"]), 0
        for idx, l in enumerate(det["lines"]):
            if idx in matched: dlok += 1; continue
            for samp, color in vis_lines:
                if l["color"]==color and np.median([_seg_dist(pt,l) for pt in samp]) <= 15:
                    dlok += 1; break
        pdots = toimg(MODEL_DOTS)
        vis = [p for p in pdots if 0<=p[0]<w and 0<=p[1]<h]
        dv = len(vis)
        dh = sum(1 for p in vis if any(np.hypot(p[0]-x,p[1]-y)<=18 for x,y in det["dots"]))
        dd = len(det["dots"])
        ddok = sum(1 for x,y in det["dots"] if any(np.hypot(p[0]-x,p[1]-y)<=18 for p in vis))
        for k,v in zip(("line_vis","line_hit","det_lines","det_line_ok",
                        "dot_vis","dot_hit","det_dots","det_dot_ok"),
                       (lv,lh,dl,dlok,dv,dh,dd,ddok)): tot[k]+=v
        print(f"{base:26s} {dl:5d}/{lh}/{lv:<8d} {dd:5d}/{dh}/{dv:<8d}")
    def pct(a,b): return f"{100*a/b:5.1f}%" if b else "  n/a"
    print("\n=== AGGREGATE (occluded features count as misses -> recall is a lower bound) ===")
    print(f"LINE recall {pct(tot['line_hit'],tot['line_vis'])}   precision {pct(tot['det_line_ok'],tot['det_lines'])}")
    print(f"DOT  recall {pct(tot['dot_hit'],tot['dot_vis'])}   precision {pct(tot['det_dot_ok'],tot['det_dots'])}")
    print("go/no-go: line recall >~75% & precision >~60% -> classical solve. else tune/escalate.")

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "show": show(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "tune": tune(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "score": score(sys.argv[2], sys.argv[3])
    else: print(__doc__)
